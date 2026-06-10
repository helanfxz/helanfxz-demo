"""
视频生成工作流。

这个模块负责把"已创建的视频任务"推进成一个可审查的视频草稿计划。
当前先不用 LangChain / LangGraph，而是用普通函数把每个步骤写清楚。
后续如果出现复杂分支、人工干预、失败重试和检查点，再迁移到工作流框架。
"""

from __future__ import annotations

import base64
import builtins
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from io import BytesIO
import json
import mimetypes
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib import request
from urllib.error import HTTPError, URLError

from agent.asset_preprocessor import create_studio_background, preprocess_all_assets
from agent.seedance_video_renderer import render_seedance_video, repair_and_rerender_shot
from agent.simple_video_renderer import render_preview_video
from agent.value_proof_planner import build_value_proof_plan, ensure_value_proof_plan

STORYBOARD_MIN_SHOTS = 3
STORYBOARD_MAX_SHOTS = 7
DEFAULT_SUBTITLE_MAX_CHARS = 24
DEFAULT_VOICEOVER_MAX_CHARS = 56

VERBOSE_LOG = os.getenv("AIGC_VERBOSE_LOG") == "1"
PROMPT_SKILL_LIBRARY_DIR = Path(__file__).resolve().parents[1] / "prompt_skill_library"
_PROMPT_SKILL_TEMPLATE_CACHE: dict[str, str] = {}
_PROMPT_SKILL_REFERENCE_CACHE: dict[str, str] = {}
_PROMPT_SKILL_PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}|\{\s*([a-zA-Z0-9_]+)\s*\}")


def print(*args, **kwargs):  # type: ignore[override]
    """默认隐藏工作流内部细节输出。"""

    if VERBOSE_LOG:
        builtins.print(*args, **kwargs)


def _flow_print(message: str) -> None:
    """输出工作流关键阶段。"""

    builtins.print(message, flush=True)


def run_video_generation_workflow(
    task_data: dict[str, Any],
    progress_callback: Callable[[str, str, int, dict | None], None] | None = None,
    stop_after_plan_review: bool = False,
) -> dict[str, Any]:
    """
    执行从素材处理到草稿计划生成的最小工作流。

    输入使用普通字典，是为了直接兼容当前 `VideoTask.to_dict()` 的输出。
    输出也保持为普通字典，方便页面展示，后续再决定是否沉淀成数据库结构。
    """

    _load_local_env()
    task_id = str(task_data.get("task_id", "unknown"))
    workflow_started_at = time.perf_counter()
    _flow_print(f"[video_generation_workflow] 工作流开始：task_id={task_id}")

    def report(stage: str, message: str, progress: int, partial: dict | None = None) -> None:
        """把工作流内部进度通知给任务模块；没有回调时保持原来的同步调用方式。"""

        if progress_callback:
            progress_callback(stage, message, progress, partial)

    steps: list[dict[str, str]] = []

    # Phase 0: 素材分析和需求结构化并行执行。
    # process_assets 已合并素材角色分析和商品身份卡，一次多模态调用完成。
    report("asset_analysis", "正在分析上传素材并生成商品身份卡。", 10)
    report("requirement_structurization", "正在整理和结构化用户需求。", 10)
    phase0_started_at = time.perf_counter()

    with ThreadPoolExecutor(max_workers=2) as pool:
        asset_future = pool.submit(process_assets, task_data)
        struct_future = pool.submit(structurize_user_requirements, task_data)
        asset_analysis = asset_future.result()
        structured_requirements = _safe_dict(struct_future.result())

    task_data["structured_requirements"] = structured_requirements
    product_identity_card = asset_analysis.get("product_identity_card", {})
    steps.append(_step("asset_analysis", "completed", "素材规则检查和语义分析完成。"))
    steps.append(_step("requirement_structurization", "completed", "用户需求结构化完成。"))
    _flow_print(
        "[video_generation_workflow] Phase 0 并行完成："
        f"elapsed={_elapsed_seconds(phase0_started_at)}s"
    )

    # 1. 商品上下文：纯计算，整合所有已有信息为后续创作节点提供统一输入。
    product_context = build_product_context(task_data, asset_analysis)
    product_context["structured_requirements"] = structured_requirements
    asset_capability_plan = build_asset_capability_plan(asset_analysis, product_context)
    product_context["asset_capability_plan"] = asset_capability_plan
    asset_analysis["asset_capability_plan"] = asset_capability_plan
    steps.append(_step("asset_capability", "completed", "素材能力评估完成，已生成可支持/不可支持镜头约束。"))
    input_confidence = task_data.get("input_confidence", "") or structured_requirements.get("input_confidence", "medium")
    product_context["input_confidence"] = input_confidence
    if input_confidence == "low":
        product_context["conservative_constraints"] = _apply_conservative_strategy(product_context, product_identity_card)
    steps.append(_step("product_context", "completed", "商品上下文整理完成。"))

    # 2 & 3. 创作规划：根据商品身份信心选择路径。
    #   路径B（模板+1LLM）：identity_confidence=high/medium，品牌信息直传，消除信息递减。
    #   路径A（三跳LLM）：identity_confidence=low，保守兜底。
    identity_confidence = product_identity_card.get("identity_confidence", "medium")
    has_appearance_anchor = bool(
        product_identity_card.get("appearance_anchor_available")
        or _find_best_appearance_anchor(asset_analysis)
    )
    use_template_path = (
        identity_confidence in ("high", "medium")
        or not identity_confidence
        or has_appearance_anchor
    )

    if use_template_path:
        report("script_plan", "正在使用模板路径生成分镜（品牌信息直传，避免信息递减）。", 35)
        stage_started_at = time.perf_counter()
        storyboard, script_plan = plan_storyboard_from_template(product_context, asset_analysis)
        _print_stage_elapsed(task_id, "模板分镜生成", stage_started_at)
        script_review = {"passed": True, "issues": [], "source": "template_path_b"}
        storyboard_review = {"passed": True, "issues": [], "source": "template_path_b"}
        steps.append(_step("script_plan", "completed", "模板路径：剧本和分镜一次生成完成（路径B）。"))
        steps.append(_step("director_storyboard", "completed", "模板路径：分镜已包含素材绑定和渲染策略。"))
        script_attempts = 1
        storyboard_attempts = 1
        report("director_storyboard", "分镜已生成（模板路径）", 58, {"storyboard": storyboard, "workflow_steps": list(steps)})
    else:
        # 路径A：三跳LLM，identity_confidence=low 时的兜底路径
        report("script_plan", "正在生成带货剧本，并进行基础规则审核。", 30)
        stage_started_at = time.perf_counter()
        script_plan, script_review, script_attempts = _run_step_with_review(
            step_name="script_plan",
            generate_func=lambda previous_issues: plan_script(
                product_context,
                _director_context_for_script(product_context, asset_analysis),
                previous_issues,
            ),
            review_func=lambda result: review_script_plan(result),
            max_retries=1,
        )
        _print_stage_elapsed(task_id, "剧本规划", stage_started_at)
        steps.append(_step("script_plan", _review_step_status(script_review), "剧本规划和审核完成。"))
        report("script_plan", "剧本已生成", 42, {"script_plan": script_plan, "workflow_steps": list(steps)})

        report("director_storyboard", "正在生成导演策略并拆分为拍摄分镜，绑定素材。", 48)
        stage_started_at = time.perf_counter()
        storyboard, storyboard_review, storyboard_attempts = _run_step_with_review(
            step_name="director_storyboard",
            generate_func=lambda previous_issues: plan_director_storyboard(
                product_context,
                script_plan,
                asset_analysis,
                previous_issues,
            ),
            review_func=lambda result: review_storyboard(result, product_context),
            max_retries=1,
        )
        _print_stage_elapsed(task_id, "导演+分镜", stage_started_at)
        steps.append(_step("director_storyboard", _review_step_status(storyboard_review), "导演分镜和素材绑定完成。"))
        report("director_storyboard", "分镜已生成", 58, {"storyboard": storyboard, "workflow_steps": list(steps)})

    # 3.5 生成前可拍性检查：把复杂动作、商品无素材、text_to_video 展示商品等问题在渲染前修掉。
    report("shootability_review", "正在进行生成前可拍性检查，降低商品漂移和物理错误风险。", 60)
    stage_started_at = time.perf_counter()
    shootability_review = review_storyboard_shootability(storyboard, product_context, asset_analysis)
    if not shootability_review.get("passed"):
        storyboard = repair_storyboard_by_shootability(storyboard, shootability_review, product_context, asset_analysis)
        storyboard = _ensure_storyboard_fields(storyboard, product_context)
        storyboard = _enforce_storyboard_continuity_groups(storyboard)
        storyboard_review = review_storyboard(storyboard, product_context)
        shootability_review = review_storyboard_shootability(storyboard, product_context, asset_analysis)
        steps.append(_step("shootability_review", "needs_review" if not shootability_review.get("passed") else "completed", "已按可拍性规则改写高风险镜头。"))
    else:
        storyboard = _enforce_storyboard_continuity_groups(storyboard)
        steps.append(_step("shootability_review", "completed", "生成前可拍性检查通过。"))
    _print_stage_elapsed(task_id, "可拍性检查", stage_started_at)
    report("shootability_review", "可拍性检查完成", 61, {"shootability_review": shootability_review, "storyboard": storyboard, "workflow_steps": list(steps)})

    # 4. 叙事闭环审核：纯规则检查，不调 LLM。
    report("narrative_review", "正在审核叙事闭环完整性。", 56)
    stage_started_at = time.perf_counter()
    narrative_review = _rule_based_narrative_review(product_context, script_plan, storyboard)
    _print_stage_elapsed(task_id, "叙事闭环审核", stage_started_at)

    narrative_review_attempts: list[dict[str, Any]] = [
        {"stage": "initial", "review": narrative_review}
    ]

    if not narrative_review["passed"]:
        script_plan = _repair_script_by_rules(script_plan, narrative_review, product_context)
        storyboard = _repair_storyboard_by_rules(storyboard, narrative_review, script_plan, product_context)
        narrative_review = _rule_based_narrative_review(product_context, script_plan, storyboard)
        narrative_review_attempts.append({"stage": "rule_repair", "review": narrative_review})
        if not narrative_review["passed"]:
            strategy_family = _plan_strategy_family(script_plan, storyboard)
            if strategy_family != "legacy":
                narrative_review = {
                    **narrative_review,
                    "passed": False,
                    "blocked_downgrade": True,
                    "strategy_family": strategy_family,
                    "message": "策略计划未通过审核，但已保留原策略，避免静默降级成旧保守模板。",
                }
                steps.append(_step("narrative_review", "needs_review", "策略叙事审核未通过，已保留原策略等待人工确认。"))
            else:
                script_plan = _fallback_conservative_script(product_context, {})
                script_plan["narrative_downgrade"] = True
                script_plan["_fallback_reason"] = "legacy_narrative_review_failed"
                storyboard = _fallback_conservative_storyboard(product_context, script_plan)
                storyboard = _ensure_storyboard_fields(storyboard, product_context)
                # 降级后强制绑定素材，确保降级分镜也有 asset_id
                storyboard = _bind_fallback_assets(storyboard, asset_analysis)
                script_review = review_script_plan(script_plan)
                storyboard_review = review_storyboard(storyboard, product_context)
                narrative_review = {"passed": True, "issues": [], "retry_target": "", "retryable": False}
                narrative_review_attempts.append({"stage": "legacy_fallback", "review": narrative_review})
                steps.append(_step("narrative_review", "needs_review", "叙事闭环审核未通过，已降级为保守结构。"))
        else:
            steps.append(_step("narrative_review", "completed", "叙事闭环审核通过（规则修复后）。"))
    else:
        steps.append(_step("narrative_review", "completed", "叙事闭环审核通过。"))

    # 5. 素材匹配：为每个镜头绑定上传素材；素材不足时标记为后续生成。
    report("asset_matching", "正在为每个分镜匹配可用素材。", 62)
    stage_started_at = time.perf_counter()
    asset_matching = match_assets_to_storyboard(storyboard, asset_analysis)
    _print_stage_elapsed(task_id, "素材匹配", stage_started_at)
    steps.append(_step("asset_matching", "completed", "分镜素材匹配完成。"))
    report("asset_gap_completion", "正在检查分镜素材缺口，并选择可执行的补全方式。", 66)
    gap_started_at = time.perf_counter()
    asset_gap_completion = complete_asset_gaps(
        storyboard=storyboard,
        asset_matching=asset_matching,
        asset_analysis=asset_analysis,
        product_identity_card=product_context.get("product_identity_card", {}),
    )
    asset_matching = asset_gap_completion["asset_matching"]
    _print_stage_elapsed(task_id, "素材缺口补全", gap_started_at)
    steps.append(_step("asset_gap_completion", "completed", "分镜素材缺口补全决策完成。"))

    # 6. 创作计划：把分镜和素材匹配结果转换成视频渲染模块能消费的计划。
    report("creation_plan", "正在生成视频创作计划。", 70)
    stage_started_at = time.perf_counter()
    creation_plan = build_creation_plan(product_context, storyboard, asset_matching)
    _print_stage_elapsed(task_id, "创作计划", stage_started_at)
    steps.append(_step("creation_plan", "completed", "创作执行计划生成完成。"))

    if stop_after_plan_review:
        workflow_result = _build_script_review_workflow_result(
            task_id=task_id,
            task_data=task_data,
            output_dir=_task_output_dir(task_data, task_id),
            workflow_started_at=workflow_started_at,
            steps=steps,
            asset_analysis=asset_analysis,
            product_context=product_context,
            script_plan=script_plan,
            script_review=script_review,
            storyboard=storyboard,
            storyboard_review=storyboard_review,
            review_attempts=script_attempts + storyboard_attempts,
            asset_matching=asset_matching,
            asset_gap_completion=asset_gap_completion,
            creation_plan=creation_plan,
            narrative_review=narrative_review,
            narrative_review_attempts=narrative_review_attempts,
            shootability_review=shootability_review,
        )
        report(
            "script_review",
            "剧本和分镜已生成，请确认、编辑或提出修改意见后继续。",
            72,
            {
                "script_plan": script_plan,
                "storyboard": storyboard,
                "readable_script": workflow_result["readable_script"],
                "script_review_variants": workflow_result["script_review_variants"],
                "workflow_steps": list(workflow_result["workflow_steps"]),
            },
        )
        return workflow_result

    # 7. 视频渲染：优先调用 Seedance；失败时回退到本地预览。
    report("render_video", "正在调用视频模型生成分镜视频，这一步通常耗时最长。", 78)
    stage_started_at = time.perf_counter()
    render_result = render_seedance_video(
        task_id=task_id,
        creation_plan=creation_plan,
        output_dir=_task_output_dir(task_data, task_id),
    )
    if not render_result["success"]:
        fallback_result = render_preview_video(
            task_id=task_id,
            storyboard=storyboard,
            asset_matching=asset_matching,
            output_dir=_task_output_dir(task_data, task_id),
        )
        fallback_result["fallback_from"] = render_result
        _flow_print(f"[video_generation_workflow] 视频模型失败，已回退本地预览：{render_result.get('error')}")
        render_result = fallback_result

    _print_stage_elapsed(task_id, "视频渲染", stage_started_at)
    steps.append(_step("render_video", "completed" if render_result["success"] else "needs_review", "视频渲染完成。"))
    report("render_video", "视频渲染完成", 82, {"render_result": render_result, "workflow_steps": list(steps)})

    # 7.5 A/B 候选：默认 A 不变，B 单独输出理想带货场景，供人工对比和诊断。
    report("ab_variant", "正在生成理想带货场景候选版本。", 84)
    ab_variants = _render_ab_variants(
        task_id=task_id,
        task_data=task_data,
        product_context=product_context,
        asset_analysis=asset_analysis,
        script_review_variants={},
    )
    if ab_variants:
        steps.append(_step("ab_variant", "completed", "A/B 候选视频已生成，默认结果仍保持保守版。"))
    else:
        steps.append(_step("ab_variant", "needs_review", "B 候选未生成或被跳过，默认保守版不受影响。"))

    # 8. 内容审视：抽取成片关键帧，交给多模态模型检查主体一致性。
    report("content_review", "正在抽取视频关键帧，并检查商品主体是否和上传素材一致。", 88)
    stage_started_at = time.perf_counter()
    content_review = review_rendered_video_content(
        product_context=product_context,
        creation_plan=creation_plan,
        render_result=render_result,
        output_dir=_task_output_dir(task_data, task_id),
    )
    _print_stage_elapsed(task_id, "内容审视", stage_started_at)
    steps.append(
        _step(
            "content_review",
            "completed" if content_review.get("passed") else "needs_review",
            "内容级审视完成。",
        )
    )

    if content_review.get("passed") is False:
        failed_count = sum(1 for r in content_review.get("shot_reviews", []) if not r.get("pass"))
        repair_records = content_review.get("repair_records", [])
        repair_records, repair_policy = _select_auto_repair_records(
            repair_records=repair_records,
            failed_count=failed_count,
            total_shots=len(creation_plan.get("shots", [])),
        )
        content_review["repair_policy"] = repair_policy
        if repair_records:
            repair_execution = _repair_rendered_content(
                task_id=task_id,
                repair_records=repair_records,
                creation_plan=creation_plan,
                render_result=render_result,
                output_dir=_task_output_dir(task_data, task_id),
                report=report,
            )
        else:
            repair_execution = _skipped_repair_execution(repair_policy)
            _flow_print(
                "[video_generation_workflow] 自动修复已跳过："
                f"{repair_policy.get('reason', 'repair_policy_blocked')}"
            )
        content_review["repair_execution"] = repair_execution

        # 第2轮：只对实际修复成功的分镜重新审视
        repair_result_records = repair_execution.get("repair_records") or repair_execution.get("records", [])
        repaired_shot_indices = [r.get("shot_index") for r in repair_result_records if r.get("status") == "succeeded"]
        if repaired_shot_indices:
            content_review_r2 = review_rendered_video_content(
                product_context=product_context,
                creation_plan=creation_plan,
                render_result=render_result,
                output_dir=_task_output_dir(task_data, task_id),
                shot_indices_filter=repaired_shot_indices,
            )
            content_review["repair_verification"] = content_review_r2
            still_failed = sum(1 for r in content_review_r2.get("shot_reviews", []) if not r.get("pass"))
            steps.append(_step(
                "content_review",
                "needs_review" if still_failed else "completed",
                (
                    f"内容审视发现 {failed_count} 个分镜需优化；"
                    f"修复后验证：{len(repaired_shot_indices) - still_failed} 个通过，"
                    f"{still_failed} 个仍需人工确认。"
                ),
            ))
        else:
            steps.append(_step(
                "content_review",
                "needs_review",
                (
                    f"内容审视发现 {failed_count} 个分镜需优化；"
                    f"实际修复成功 {repair_execution.get('succeeded_count', 0)} 个，"
                    f"失败 {repair_execution.get('failed_count', 0)} 个，请人工确认。"
                ),
            ))
    # 修复完成后，把最新视频地址推给前端（覆盖 render_video 阶段的旧地址）
    repaired_video_url = None
    if content_review.get("repair_execution", {}).get("reconcat_success"):
        repaired_video_url = f"/uploads/{task_id}/seedance_final.mp4"
    report("content_review", "内容审视完成", 90, {
        "content_review": content_review,
        "workflow_steps": list(steps),
        "repaired_video_url": repaired_video_url,
    })

    # 9. 最终检查：用规则检查时长、分镜字段、视频产物和内容审视，决定任务是否可以进入确认。
    report("final_check", "正在进行最终检查并整理结果。", 92)
    stage_started_at = time.perf_counter()
    final_check = run_final_check(
        product_context,
        storyboard,
        creation_plan,
        render_result,
        asset_gap_completion,
        content_review,
    )
    _print_stage_elapsed(task_id, "最终检查", stage_started_at)
    critical_passed = (
        script_review["passed"]
        and storyboard_review["passed"]
        and narrative_review.get("passed", True)
        and final_check["passed"]
        and render_result["success"]
    )
    workflow_status = "completed" if critical_passed else "needs_review"
    workflow_stage = "draft_ready" if final_check["passed"] else "draft_needs_review"
    workflow_message = (
        "视频草稿计划已生成，可以进入用户确认。"
        if final_check["passed"]
        else "视频草稿计划已生成，但存在需要人工确认的问题。"
    )
    steps.append(_step("final_check", workflow_status, workflow_message))
    report(workflow_stage, workflow_message, 98, {"final_check": final_check, "workflow_steps": list(steps)})
    trace_summary = _build_trace_summary(
        asset_analysis=asset_analysis,
        script_plan=script_plan,
        review_attempts=script_attempts + storyboard_attempts,
        render_result=render_result,
        final_check=final_check,
        asset_gap_completion=asset_gap_completion,
        content_review=content_review,
        storyboard=storyboard,
    )
    trace_summary["narrative_review_passed"] = bool(narrative_review.get("passed", True))
    trace_summary["narrative_downgrade"] = bool(script_plan.get("narrative_downgrade"))
    trace_summary["narrative_downgrade_blocked"] = bool(narrative_review.get("blocked_downgrade"))
    trace_summary["strategy_family"] = _plan_strategy_family(script_plan, storyboard)
    trace_summary["ab_variant_count"] = len(ab_variants)
    trace_summary["ab_variants"] = {
        variant_id: {
            "success": bool((variant.get("render_result") or {}).get("success")),
            "video_path": variant.get("video_path", ""),
            "strategy": variant.get("strategy", ""),
        }
        for variant_id, variant in ab_variants.items()
    }

    _flow_print(
        "[video_generation_workflow] 工作流执行结束："
        f"task_id={task_id}, workflow_status={workflow_status}, "
        f"workflow_stage={workflow_stage}, total_elapsed={_elapsed_seconds(workflow_started_at)}s",
    )

    workflow_result: dict[str, Any] = {
        "workflow_status": workflow_status,
        "workflow_stage": workflow_stage,
        "workflow_message": workflow_message,
        "workflow_steps": steps,
        "trace_summary": trace_summary,
        "director_decision": {
            # 从合并后的导演+分镜产物中提取导演层信息，保持前端兼容性。
            "selected_strategy": "创意分镜组合",
            "selected_reason": f"基于{len(storyboard)}个分镜的动态组合",
            "factor_combination": {
                "narrative_framework": _guess_framework_from_storyboard(storyboard),
                "camera": _guess_camera_style_from_storyboard(storyboard),
                "pacing": _guess_pacing_from_storyboard(storyboard),
            },
            "asset_advice": [
                f"分镜{shot.get('shot_index','')}绑定素材 {shot.get('asset_id','无')} (策略:{shot.get('render_strategy','text_to_video')})"
                for shot in storyboard if shot.get("asset_id")
            ] or ["所有分镜均使用文本生成视频"],
            "render_advice": "image_to_video" if any(s.get("render_strategy") == "image_to_video" for s in storyboard) else "text_to_video",
            "candidate_variants": [],
        },
        "asset_analysis": asset_analysis,
        "state_transition_schema": {
            "narrative_role": "分镜叙事角色，例如 hook / feature_demo / detail_proof / cta",
            "scene_goal": "这个镜头要完成的表达目标",
            "initial_state": "镜头开始时的画面状态",
            "action": "镜头中发生的核心动作或变化",
            "final_state": "镜头结束时的画面状态",
            "camera_motion": "镜头运动方式，例如推近、横移、定镜",
            "asset_usage": "该镜头如何使用素材，包括 usage_type、required_asset_role、is_identity_critical",
            "product_identity_constraints": "从商品身份卡继承的必须保持项",
        },
        "product_identity_card": product_context.get("product_identity_card", {}),
        "motion_affordance": product_context.get("motion_affordance", {}),
        "asset_profiles": product_context.get("asset_profiles", []),
        "product_context": _product_context_for_llm(product_context),
        "script_plan": script_plan,
        "script_review": script_review,
        "storyboard": storyboard,
        "storyboard_review": storyboard_review,
        "review_attempts": script_attempts + storyboard_attempts,
        "asset_matching": asset_matching,
        "asset_gap_completion": asset_gap_completion,
        "creation_plan": creation_plan,
        "render_result": render_result,
        "ab_variants": ab_variants,
        "content_review": content_review,
        "narrative_review": narrative_review,
        "narrative_review_attempts": narrative_review_attempts,
        "shootability_review": shootability_review,
        "final_check": final_check,
    }

    artifacts_dir = _save_workflow_artifacts(
        task_id=task_id,
        output_dir=_task_output_dir(task_data, task_id),
        artifacts={
            **workflow_result,
            "product_identity_card": product_context.get("product_identity_card", {}),
            "product_context": product_context,
            "structured_requirements": task_data.get("structured_requirements", {}),
        },
    )
    workflow_result["artifacts_dir"] = artifacts_dir
    builtins.print(f"[video_generation_workflow] 中间产物已保存：{artifacts_dir}", flush=True)

    return workflow_result


def continue_video_generation_workflow(
    task_data: dict[str, Any],
    progress_callback: Callable[[str, str, int, dict | None], None] | None = None,
) -> dict[str, Any]:
    """从用户确认/编辑后的剧本分镜继续执行渲染和审核阶段。"""

    _load_local_env()
    task_id = str(task_data.get("task_id", "unknown"))
    workflow_started_at = time.perf_counter()
    previous = _safe_dict(task_data.get("workflow_result", {}))
    if not previous.get("script_plan") or not previous.get("storyboard"):
        raise ValueError("缺少已确认的剧本或分镜，无法继续渲染。")

    def report(stage: str, message: str, progress: int, partial: dict | None = None) -> None:
        if progress_callback:
            progress_callback(stage, message, progress, partial)

    steps = list(previous.get("workflow_steps", []))
    steps.append(_step("script_review", "completed", "用户已确认剧本和分镜，继续渲染。"))
    asset_analysis = _safe_dict(previous.get("asset_analysis", {}))
    product_context = _safe_dict(previous.get("product_context", {}))
    if not product_context:
        product_context = build_product_context(task_data, asset_analysis)
    script_plan = _safe_dict(previous.get("script_plan", {}))
    storyboard = _normalize_storyboard(previous.get("storyboard", []))
    script_review = _safe_dict(previous.get("script_review", {"passed": True, "issues": []}))
    storyboard_review = _safe_dict(previous.get("storyboard_review", {"passed": True, "issues": []}))
    narrative_review = _safe_dict(previous.get("narrative_review", {"passed": True, "issues": []}))
    narrative_review_attempts = list(previous.get("narrative_review_attempts", []))
    shootability_review = _safe_dict(previous.get("shootability_review", {"passed": True, "issues": []}))
    review_attempts = previous.get("review_attempts", 1)
    if isinstance(review_attempts, list):
        review_attempt_count = len(review_attempts)
    else:
        review_attempt_count = int(review_attempts or 1)

    report("asset_matching", "正在按确认后的分镜重新匹配素材。", 74)
    asset_matching = match_assets_to_storyboard(storyboard, asset_analysis)
    asset_gap_completion = complete_asset_gaps(
        storyboard=storyboard,
        asset_matching=asset_matching,
        asset_analysis=asset_analysis,
        product_identity_card=product_context.get("product_identity_card", {}),
    )
    asset_matching = asset_gap_completion.get("asset_matching", asset_matching)
    creation_plan = build_creation_plan(product_context, storyboard, asset_matching)
    steps.append(_step("creation_plan", "completed", "已按确认后的分镜生成创作执行计划。"))

    report("render_video", "正在调用视频模型生成分镜视频，这一步通常耗时最长。", 78)
    render_started_at = time.perf_counter()
    render_result = render_seedance_video(
        task_id=task_id,
        creation_plan=creation_plan,
        output_dir=_task_output_dir(task_data, task_id),
    )
    if not render_result["success"]:
        fallback_result = render_preview_video(
            task_id=task_id,
            storyboard=storyboard,
            asset_matching=asset_matching,
            output_dir=_task_output_dir(task_data, task_id),
        )
        fallback_result["fallback_from"] = render_result
        render_result = fallback_result
    _print_stage_elapsed(task_id, "用户确认后视频渲染", render_started_at)
    steps.append(_step("render_video", "completed" if render_result["success"] else "needs_review", "视频渲染完成。"))
    report("render_video", "视频渲染完成", 84, {"render_result": render_result, "workflow_steps": list(steps)})

    report("ab_variant", "正在生成理想带货场景候选版本。", 86)
    ab_variants = _render_ab_variants(
        task_id=task_id,
        task_data=task_data,
        product_context=product_context,
        asset_analysis=asset_analysis,
        script_review_variants=_safe_dict(previous.get("script_review_variants", {})),
    )
    steps.append(_step("ab_variant", "completed" if ab_variants else "needs_review", "A/B 候选处理完成。"))

    report("content_review", "正在抽取视频关键帧，并检查商品主体是否和上传素材一致。", 90)
    content_review = review_rendered_video_content(
        product_context=product_context,
        creation_plan=creation_plan,
        render_result=render_result,
        output_dir=_task_output_dir(task_data, task_id),
    )
    steps.append(_step("content_review", "completed" if content_review.get("passed") else "needs_review", "内容级审视完成。"))
    report("content_review", "内容审视完成", 92, {"content_review": content_review, "workflow_steps": list(steps)})

    report("final_check", "正在进行最终检查并整理结果。", 96)
    final_check = run_final_check(
        product_context,
        storyboard,
        creation_plan,
        render_result,
        asset_gap_completion,
        content_review,
    )
    critical_passed = (
        script_review.get("passed", True)
        and storyboard_review.get("passed", True)
        and narrative_review.get("passed", True)
        and final_check.get("passed", False)
        and render_result.get("success", False)
    )
    workflow_status = "completed" if critical_passed else "needs_review"
    workflow_stage = "draft_ready" if final_check.get("passed") else "draft_needs_review"
    workflow_message = (
        "视频已生成，可对比 A/B 版本并选择可用结果。"
        if critical_passed
        else "视频已生成，但存在需要人工确认的问题。"
    )
    steps.append(_step("final_check", workflow_status, workflow_message))

    trace_summary = _build_trace_summary(
        asset_analysis=asset_analysis,
        script_plan=script_plan,
        review_attempts=review_attempt_count,
        render_result=render_result,
        final_check=final_check,
        asset_gap_completion=asset_gap_completion,
        content_review=content_review,
        storyboard=storyboard,
    )
    trace_summary["ab_variant_count"] = len(ab_variants)
    trace_summary["strategy_family"] = _plan_strategy_family(script_plan, storyboard)

    workflow_result = {
        **previous,
        "workflow_status": workflow_status,
        "workflow_stage": workflow_stage,
        "workflow_message": workflow_message,
        "workflow_progress": 100,
        "workflow_steps": steps,
        "trace_summary": trace_summary,
        "asset_analysis": asset_analysis,
        "product_context": product_context,
        "script_plan": script_plan,
        "script_review": script_review,
        "storyboard": storyboard,
        "storyboard_review": storyboard_review,
        "review_attempts": review_attempt_count,
        "asset_matching": asset_matching,
        "asset_gap_completion": asset_gap_completion,
        "creation_plan": creation_plan,
        "render_result": render_result,
        "ab_variants": ab_variants,
        "content_review": content_review,
        "narrative_review": narrative_review,
        "narrative_review_attempts": narrative_review_attempts,
        "shootability_review": shootability_review,
        "final_check": final_check,
    }
    artifacts_dir = _save_workflow_artifacts(
        task_id=task_id,
        output_dir=_task_output_dir(task_data, task_id),
        artifacts=workflow_result,
    )
    workflow_result["artifacts_dir"] = artifacts_dir
    _flow_print(
        "[video_generation_workflow] 用户确认后工作流执行结束："
        f"task_id={task_id}, workflow_status={workflow_status}, "
        f"workflow_stage={workflow_stage}, total_elapsed={_elapsed_seconds(workflow_started_at)}s"
    )
    return workflow_result


def _build_script_review_workflow_result(
    *,
    task_id: str,
    task_data: dict[str, Any],
    output_dir: str,
    workflow_started_at: float,
    steps: list[dict[str, str]],
    asset_analysis: dict[str, Any],
    product_context: dict[str, Any],
    script_plan: dict[str, Any],
    script_review: dict[str, Any],
    storyboard: list[dict[str, Any]],
    storyboard_review: dict[str, Any],
    review_attempts: int,
    asset_matching: list[dict[str, Any]],
    asset_gap_completion: dict[str, Any],
    creation_plan: dict[str, Any],
    narrative_review: dict[str, Any],
    narrative_review_attempts: list[dict[str, Any]],
    shootability_review: dict[str, Any],
) -> dict[str, Any]:
    """整理给前端人工审阅的剧本/分镜草稿，不触发视频渲染。"""

    review_steps = list(steps)
    review_steps.append(_step("script_review", "needs_review", "剧本和分镜等待用户确认。"))
    script_plan = dict(script_plan or {})
    if not str(script_plan.get("rich_story_text", "")).strip():
        product_type = str(
            product_context.get("product_type")
            or product_context.get("product_identity_card", {}).get("product_type")
            or script_plan.get("grounded_product_type")
            or "商品"
        ).strip()
        script_plan["rich_story_text"] = _rich_story_text_from_product_context(
            storyboard,
            product_type,
            product_context,
        )
    script_review_variants = _build_script_review_variants(
        product_context=product_context,
        asset_analysis=asset_analysis,
        script_plan=script_plan,
        storyboard=storyboard,
    )
    workflow_result = {
        "workflow_status": "needs_review",
        "workflow_stage": "script_review",
        "workflow_message": "剧本和分镜已生成，请确认、编辑或提出修改意见后继续。",
        "workflow_progress": 72,
        "workflow_steps": review_steps,
        "trace_summary": {
            "review_attempt_count": review_attempts,
            "render_mode": "pending_user_approval",
            "final_check_passed": False,
        },
        "asset_analysis": asset_analysis,
        "product_context": _product_context_for_llm(product_context),
        "product_identity_card": product_context.get("product_identity_card", {}),
        "script_plan": script_plan,
        "script_review": script_review,
        "storyboard": storyboard,
        "storyboard_review": storyboard_review,
        "review_attempts": review_attempts,
        "asset_matching": asset_matching,
        "asset_gap_completion": asset_gap_completion,
        "creation_plan": creation_plan,
        "render_result": {},
        "ab_variants": {},
        "script_review_variants": script_review_variants,
        "content_review": {},
        "narrative_review": narrative_review,
        "narrative_review_attempts": narrative_review_attempts,
        "shootability_review": shootability_review,
        "final_check": {},
        "readable_script": _readable_script_for_review(script_plan, storyboard),
        "elapsed_seconds": round(time.perf_counter() - workflow_started_at, 2),
    }
    artifacts_dir = _save_workflow_artifacts(
        task_id=task_id,
        output_dir=output_dir,
        artifacts={
            **workflow_result,
            "product_context": product_context,
            "structured_requirements": task_data.get("structured_requirements", {}),
        },
    )
    workflow_result["artifacts_dir"] = artifacts_dir
    return workflow_result


def _build_script_review_variants(
    *,
    product_context: dict[str, Any],
    asset_analysis: dict[str, Any],
    script_plan: dict[str, Any],
    storyboard: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build front-end script review variants without rendering videos."""

    variants: dict[str, Any] = {
        "A_conservative_fidelity": _script_review_variant_payload(
            label="A 保守保真版",
            description="沿用当前工作流生成的剧本和分镜，优先保持商品身份、素材锚点和可拍性稳定。",
            script_plan=script_plan,
            storyboard=storyboard,
        )
    }

    try:
        b_storyboard = _plan_ideal_commerce_scene_storyboard(product_context, asset_analysis)
    except Exception as exc:  # pragma: no cover - defensive guard for review UX
        _flow_print(f"[video_generation_workflow] B 审阅候选生成失败，保留 A-only：{exc}")
        return variants

    if not b_storyboard:
        return variants

    try:
        identity_card = _safe_dict(product_context.get("product_identity_card", {}))
        product_type = str(
            identity_card.get("product_type")
            or product_context.get("product_type")
            or product_context.get("product_title")
            or script_plan.get("grounded_product_type")
            or "商品"
        ).strip()
        b_duration = sum(
            _safe_int(shot.get("duration_seconds"), default=0)
            for shot in b_storyboard
        ) or 15
        b_script_plan = _build_template_script_plan_stub(
            b_storyboard,
            product_type,
            "B_ideal_commerce_scene:script_review_variant",
            duration=b_duration,
            product_context=product_context,
        )
        variants["B_ideal_commerce_scene"] = _script_review_variant_payload(
            label="B 理想带货场景版",
            description="使用理想带货单场景方案生成的候选剧本和分镜，用于在渲染前对比更大胆的商业表达上限。",
            script_plan=b_script_plan,
            storyboard=b_storyboard,
        )
    except Exception as exc:  # pragma: no cover - defensive guard for review UX
        _flow_print(f"[video_generation_workflow] B 审阅候选整理失败，保留 A-only：{exc}")
    return variants


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _script_review_variant_payload(
    *,
    label: str,
    description: str,
    script_plan: dict[str, Any],
    storyboard: list[dict[str, Any]],
) -> dict[str, Any]:
    normalized_script_plan = dict(script_plan or {})
    normalized_storyboard = [
        dict(shot)
        for shot in (storyboard or [])
        if isinstance(shot, dict)
    ]
    return {
        "label": label,
        "description": description,
        "script_plan": normalized_script_plan,
        "storyboard": normalized_storyboard,
        "readable_script": _readable_script_for_review(normalized_script_plan, normalized_storyboard),
    }


def _readable_script_for_review(script_plan: dict[str, Any], storyboard: list[dict[str, Any]]) -> dict[str, Any]:
    """把结构化剧本转成前端可直接阅读和编辑的轻量结构。"""

    synopsis = str(
        script_plan.get("rich_story_text")
        or script_plan.get("story_synopsis")
        or script_plan.get("core_message")
        or ""
    ).strip()
    if not synopsis:
        synopsis = _compose_story_synopsis_from_storyboard(storyboard)
    return {
        "synopsis": synopsis,
        "hook": str(script_plan.get("hook", "")).strip(),
        "body": _string_list(script_plan.get("body", [])),
        "cta": str(script_plan.get("cta", "")).strip(),
        "shots": [
            {
                "shot_index": shot.get("shot_index", index),
                "duration_seconds": shot.get("duration_seconds", ""),
                "role": shot.get("narrative_role") or shot.get("purpose", ""),
                "goal": shot.get("scene_goal") or shot.get("purpose", ""),
                "action": shot.get("action", ""),
                "subtitle": shot.get("subtitle", ""),
            }
            for index, shot in enumerate(storyboard, start=1)
        ],
    }


def _compose_story_synopsis_from_storyboard(storyboard: list[dict[str, Any]]) -> str:
    goals = [
        str(shot.get("scene_goal") or shot.get("purpose") or "").strip()
        for shot in storyboard
        if str(shot.get("scene_goal") or shot.get("purpose") or "").strip()
    ]
    actions = [
        str(shot.get("action") or "").strip()
        for shot in storyboard
        if str(shot.get("action") or "").strip()
    ]
    subtitles = [
        str(shot.get("subtitle") or "").strip()
        for shot in storyboard
        if str(shot.get("subtitle") or "").strip()
    ]
    if goals:
        return "这条视频先" + "，再".join(goals[:4]) + "，最后引导用户理解商品价值。"
    if actions:
        return "这条视频通过" + "，".join(actions[:4]) + "来展示商品卖点和使用结果。"
    if subtitles:
        return "这条视频围绕“" + "、".join(subtitles[:4]) + "”展开商品带货表达。"
    return "这条视频会围绕商品真实外观、核心卖点和使用场景，形成一条短视频带货故事。"


def structurize_user_requirements(task_data: dict[str, Any]) -> dict[str, Any]:
    """把表单、选项和聊天记录总结为后端可消费的结构化需求。"""

    prompt = {
        "task": "把用户的视频生成需求整理成结构化 JSON，供后续剧本、分镜和审查节点消费。",
        "instruction": "只返回 JSON，不要返回解释文字。用户没有明确说的信息不要编造。",
        "input": {
            "product_title": task_data.get("title", ""),
            "selling_points": task_data.get("selling_points", []),
            "custom_style": task_data.get("custom_style_prompt", ""),
            "product_type": task_data.get("product_type", ""),
            "target_audience": task_data.get("target_audience", ""),
            "usage_scene": task_data.get("usage_scene", ""),
            "creative_direction": task_data.get("creative_direction", ""),
            "forbidden_changes": task_data.get("forbidden_changes", []),
            "chat_history": task_data.get("chat_history", ""),
            "has_uploaded_assets": bool(task_data.get("uploaded_assets")),
        },
        "output_format": {
            "product_type": "",
            "target_audience": "",
            "usage_scene": "",
            "creative_goal": "",
            "selling_point_priority": [],
            "must_preserve": [],
            "avoid": [],
            "tone": "",
            "extra_requirements": "",
            "input_confidence": "high / medium / low",
        },
    }
    llm_result = _call_text_llm(prompt, purpose="structurize_requirements")

    if llm_result["ok"]:
        parsed = _extract_json_from_text(llm_result["content"])
        if isinstance(parsed, dict):
            return _normalize_structured_requirements(parsed, task_data)

    return _fallback_structured_requirements(task_data)


def _normalize_structured_requirements(raw: dict[str, Any], task_data: dict[str, Any]) -> dict[str, Any]:
    """归一化 LLM 返回的需求结构化结果。"""

    fallback = _fallback_structured_requirements(task_data)
    result = dict(fallback)
    for key in ["product_type", "target_audience", "usage_scene", "creative_goal", "tone", "extra_requirements"]:
        value = str(raw.get(key, "")).strip()
        if value:
            result[key] = value
    for key in ["selling_point_priority", "must_preserve", "avoid"]:
        value = raw.get(key, [])
        if isinstance(value, list) and value:
            result[key] = [str(item).strip() for item in value if str(item).strip()]
    confidence = str(raw.get("input_confidence", "")).strip().lower()
    if confidence in ("high", "medium", "low"):
        result["input_confidence"] = confidence
    return result


def _fallback_structured_requirements(task_data: dict[str, Any]) -> dict[str, Any]:
    """LLM 不可用时，根据表单字段生成基础需求结构。"""

    selling_points = [str(p).strip() for p in task_data.get("selling_points", []) if str(p).strip()]
    forbidden_changes_raw = task_data.get("forbidden_changes", [])
    if isinstance(forbidden_changes_raw, str):
        forbidden_changes_list = [c.strip() for c in forbidden_changes_raw.split(",") if c.strip()]
    else:
        forbidden_changes_list = [str(c).strip() for c in forbidden_changes_raw if str(c).strip()]
    must_preserve = ["保持商品主体形态和关键结构"]
    if forbidden_changes_list:
        must_preserve.extend(forbidden_changes_list)

    return {
        "target_audience": str(task_data.get("target_audience", "")).strip(),
        "usage_scene": str(task_data.get("usage_scene", "")).strip(),
        "product_type": str(task_data.get("product_type", "")).strip(),
        "creative_goal": str(task_data.get("creative_direction", "")).strip(),
        "selling_point_priority": selling_points,
        "must_preserve": must_preserve,
        "avoid": [],
        "tone": str(task_data.get("custom_style_prompt", "")).strip(),
        "extra_requirements": "",
        "input_confidence": str(task_data.get("input_confidence", "medium")),
    }

def process_assets(task_data: dict[str, Any]) -> dict[str, Any]:
    """处理上传素材：图像预处理 + 一次多模态调用同时完成素材角色分析和商品身份卡生成。

    之前 process_assets 和 build_product_identity_card 两次多模态调用
    传入的是同一组图片，现在合并为一次调用，同时产出两部分。
    """

    print("[video_generation_workflow] 开始素材处理。", flush=True)
    raw_assets = task_data.get("uploaded_assets", [])
    assets = [_normalize_asset(asset) for asset in raw_assets]
    supported_assets = [asset for asset in assets if asset["is_supported"]]

    # 素材预处理：统一做清晰度、曝光、白平衡修复和画幅适配。
    task_id = str(task_data.get("task_id", "unknown"))
    preprocess_output_dir = Path(_task_output_dir(task_data, task_id)) / "preprocessed"
    preprocess_output_dir.mkdir(parents=True, exist_ok=True)
    preprocess_results = preprocess_all_assets(assets, str(preprocess_output_dir))
    background_removed_count = sum(1 for result in preprocess_results if result.get("background_removed"))
    background_fallback_count = sum(
        1
        for result in preprocess_results
        if result.get("output_path") and result.get("background_removed") is False
    )
    _flow_print(
        "[video_generation_workflow] 素材预处理完成："
        f"total={len(preprocess_results)}, "
        f"sharpness_fixed={sum(1 for r in preprocess_results if r.get('sharpness_fixed'))}, "
        f"exposure_fixed={sum(1 for r in preprocess_results if r.get('exposure_fixed'))}, "
        f"background_removed={background_removed_count}, "
        f"background_fallback={background_fallback_count}"
    )
    for asset in assets:
        asset_path = asset.get("file_path", "")
        for result in preprocess_results:
            if result.get("original_path") == asset_path and result.get("output_path"):
                asset["original_file_path"] = asset_path
                asset["standardized_file_path"] = result["output_path"]
                asset["anchor_file_path"] = result.get("anchor_output_path", "")
                asset["keyframe_variants"] = result.get("keyframe_variants", {})
                # 主商品 bbox 和 mask 需要继续传给后续渲染、审视以及前端确认流程。
                asset["primary_product"] = result.get("primary_product", {})
                # 渲染优先使用统一背景锚点图；抠图失败时安全回退标准化原图。
                asset["file_path"] = result.get("anchor_output_path") or result["output_path"]
                break
    image_paths = [
        asset.get("standardized_file_path") or asset["file_path"]
        for asset in assets
        if asset["asset_type"] == "image" and asset.get("file_path") and Path(asset["file_path"]).exists()
    ]

    # 一次多模态调用同时完成素材角色分析和商品身份卡生成。
    prompt = {
        "task": "分析上传的商品图片，同时完成两项工作：1) 素材角色分析 2) 商品身份卡。",
        "method": "先看每张图的画面内容（商品主体、背景、构图），然后同时产出角色分配和身份卡。",
        "roles": {
            "hook_opener": "前 2 秒抓注意力——需要画面干净、主体突出、有视觉冲击力",
            "product_showcase": "清晰展示商品外观——需要主体完整、光线充足、无遮挡",
            "detail_closeup": "特写商品细节（材质、接口、纹理）——需要高清晰度",
            "scene_context": "展示使用场景或 lifestyle——需要场景感、自然光",
            "cta_closer": "结尾引导行动——需要干净画面留出字幕空间",
        },
        "asset_metadata": _assets_for_llm(assets),
        "filename_policy": "文件名、路径和上传链接不得作为视觉事实依据；素材理解和身份卡只能基于图片内容、商品标题和卖点。",
        "product_title": task_data.get("title", ""),
        "selling_points": task_data.get("selling_points", []),
        "output_format": {
            "asset_roles": [
                {
                    "asset_id": "素材 ID",
                    "filename": "文件名（仅用于对应，不得影响视觉判断）",
                    "suitable_for": ["hook_opener", "detail_closeup"],
                    "visual_role": "appearance_anchor / scene_context / detail_reference",
                    "reason": "中文说明为什么适合这些角色",
                    "quality_score": 80,
                    "product_visibility": "主体清晰 / 部分可见 / 不包含商品",
                    "background_type": "干净背景 / 场景背景 / 复杂背景",
                    "identity_contribution": ["商品外观约束"],
                    "risk_notes": [],
                }
            ],
            "product_identity_card": {
                "product_type": "商品类型",
                "identity_confidence": "high / medium / low",
                "appearance_summary": "稳定外观摘要——只写图片中确实可见的特征",
                "primary_color": "主色调",
                "secondary_colors": [],
                "material_features": [],
                "shape_features": [],
                "key_components": [],
                "visible_marks": [],
                "functional_features": [],
                "scale_or_size_cues": [],
                "must_preserve": [],
                "allowed_variations": [],
                "forbidden_changes": [],
                "reference_asset_ids": [],
                "motion_affordance": {
                    "can_move_by_itself": False,
                    "can_fly": False,
                    "can_rotate": "camera_orbit_only / freely / none",
                    "can_open_or_close": False,
                    "can_be_handheld": True,
                    "allowed_actions": [],
                    "forbidden_actions": [],
                },
            },
        },
        "instruction": "只返回 JSON，不要返回解释文字。外观、颜色、材质、logo 只能来自图片内容。看不清的字段写 unknown 或空数组。",
    }
    if image_paths:
        llm_result = _call_multimodal_llm(prompt, image_paths=image_paths, purpose="asset_analysis")
    else:
        llm_result = {"ok": False, "content": "", "error": "没有可读图片，无法进行多模态素材分析。"}

    # 解析 LLM 返回的合并结果。调用成功但 JSON 解析失败时，保留原文并做一次文本修复。
    raw_multimodal_response = str(llm_result.get("content", "") or "")
    vision_parse_failed = False
    vision_parse_repaired = False
    fallback_used = False
    fallback_reason = ""
    parsed: Any = None
    if llm_result["ok"]:
        parsed = _extract_json_from_text(raw_multimodal_response)
        if not isinstance(parsed, dict):
            vision_parse_failed = True
            repair_result = _repair_json_response(
                raw_multimodal_response,
                purpose="asset_analysis",
                expected_shape=prompt.get("output_format", {}),
            )
            if repair_result.get("ok") and isinstance(repair_result.get("parsed"), dict):
                parsed = repair_result["parsed"]
                vision_parse_repaired = True
            else:
                vision_parse_failed = True
                fallback_used = True
                fallback_reason = (
                    "多模态返回内容无法解析，JSON 修复也失败："
                    f"{repair_result.get('error') or 'unknown_error'}"
                )
        if isinstance(parsed, dict):
            asset_roles_raw = parsed.get("asset_roles", [])
            identity_card_raw = parsed.get("product_identity_card", {})
            semantic_summary = _format_role_summary_from_parsed(asset_roles_raw, assets)
            asset_profiles = _build_asset_profiles_from_parsed(
                asset_roles_raw,
                assets,
                role_source="multimodal_repaired" if vision_parse_repaired else "multimodal",
            )
            identity_card = _normalize_product_identity_card(identity_card_raw, task_data, {"assets": assets, "asset_profiles": asset_profiles})
            if identity_card:
                identity_card["llm_enabled"] = True
                identity_card["llm_notes"] = raw_multimodal_response
                identity_card["vision_parse_repaired"] = vision_parse_repaired
            else:
                identity_card = _fallback_product_identity_card(task_data, {"assets": assets, "asset_profiles": asset_profiles})
                identity_card["llm_enabled"] = False
                identity_card["llm_error"] = "多模态返回了内容但无法解析为身份卡 JSON。"
        else:
            semantic_summary = _fallback_asset_summary(assets, task_data)
            asset_profiles = _build_asset_profiles(assets, semantic_summary)
            identity_card = _fallback_product_identity_card(task_data, {"assets": assets, "asset_profiles": asset_profiles})
            identity_card["llm_enabled"] = False
            identity_card["identity_confidence"] = "low"
            identity_card["llm_error"] = fallback_reason or "多模态返回内容无法解析。"
    else:
        fallback_used = True
        fallback_reason = llm_result.get("error") or "多模态 LLM 调用失败。"
        semantic_summary = _fallback_asset_summary(assets, task_data)
        asset_profiles = _build_asset_profiles(assets, semantic_summary)
        identity_card = _fallback_product_identity_card(task_data, {"assets": assets, "asset_profiles": asset_profiles})
        identity_card["llm_enabled"] = False
        identity_card["identity_confidence"] = "low"
        identity_card["llm_error"] = fallback_reason

    _merge_structured_requirements_into_identity_card(identity_card, task_data, task_data.get("structured_requirements"))
    identity_card["appearance_anchor_available"] = any(_is_full_product_anchor(profile) for profile in asset_profiles)
    # 多模态画像完成后再确定渲染锚点。这样局部特写不会被误当成完整商品外观复用。
    _apply_asset_profiles_to_assets(assets, asset_profiles)
    shared_anchor_fallbacks = _apply_shared_anchor_fallback(assets)
    if shared_anchor_fallbacks:
        _flow_print(
            "[video_generation_workflow] 部分素材抠图未通过质量检查，已复用同商品合格锚点："
            f"count={shared_anchor_fallbacks}"
        )
    shared_scene_background_path = ""
    if any(asset.get("anchor_file_path") for asset in assets):
        shared_scene_background_path = create_studio_background(str(preprocess_output_dir))
    asset_selection_diagnostics = _build_asset_selection_diagnostics(
        assets=assets,
        asset_profiles=asset_profiles,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        vision_parse_failed=vision_parse_failed,
        vision_parse_repaired=vision_parse_repaired,
    )

    print(
        "[video_generation_workflow] 素材处理完成："
        f"asset_count={len(assets)}, supported_count={len(supported_assets)}, "
        f"llm_enabled={llm_result['ok']}, identity_confidence={identity_card.get('identity_confidence')}, "
        "primary_product_confirmation_required="
        f"{sum(1 for asset in assets if asset.get('primary_product', {}).get('requires_user_confirmation'))}",
        flush=True,
    )

    return {
        "asset_count": len(assets),
        "supported_count": len(supported_assets),
        "assets": assets,
        "asset_profiles": asset_profiles,
        "semantic_summary": semantic_summary,
        "llm_enabled": llm_result["ok"],
        "llm_error": llm_result.get("error"),
        "raw_response": {"asset_analysis": raw_multimodal_response} if raw_multimodal_response else {},
        "raw_multimodal_response": raw_multimodal_response,
        "raw_response_saved": bool(raw_multimodal_response),
        "vision_parse_failed": vision_parse_failed,
        "vision_parse_repaired": vision_parse_repaired,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "asset_selection_diagnostics": asset_selection_diagnostics,
        "preprocess_results": preprocess_results,
        "product_identity_card": identity_card,
        "shared_scene_background_path": shared_scene_background_path,
    }


def _apply_shared_anchor_fallback(assets: list[dict[str, Any]]) -> int:
    """抠图失败时复用同批商品的合格锚点图，避免原始照片背景进入成片。"""

    clean_anchor_asset = _best_appearance_anchor_asset(assets)
    if not clean_anchor_asset:
        return 0

    shared_anchor_path = clean_anchor_asset["anchor_file_path"]
    source_asset_id = clean_anchor_asset.get("asset_id", "")
    fallback_count = 0
    for asset in assets:
        if asset.get("asset_type") != "image" or asset.get("anchor_file_path"):
            continue
        if not asset.get("standardized_file_path"):
            continue
        # 标准化原图仍保留给多模态理解；只有渲染入口切换到统一背景锚点。
        asset["file_path"] = shared_anchor_path
        asset["render_anchor_source_asset_id"] = source_asset_id
        asset["shared_anchor_fallback"] = True
        fallback_count += 1
    return fallback_count


def _apply_asset_profiles_to_assets(
    assets: list[dict[str, Any]],
    asset_profiles: list[dict[str, Any]],
) -> None:
    """把多模态素材画像合并回素材记录，供下游选择合适的渲染锚点。"""

    profiles_by_id = {str(profile.get("asset_id", "")): profile for profile in asset_profiles}
    for asset in assets:
        profile = profiles_by_id.get(str(asset.get("asset_id", "")))
        if not profile:
            continue
        for key in (
            "visual_role",
            "quality_score",
            "product_visibility",
            "suitable_for",
            "not_suitable_for",
            "risk_notes",
            "normalized_roles",
            "material_capabilities",
            "role_source",
        ):
            asset[key] = profile.get(key)


def _best_appearance_anchor_asset(assets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """优先选择完整外观锚点；画像缺失时再回退任意可用锚点。"""

    candidates = [
        asset
        for asset in assets
        if asset.get("asset_type") == "image"
        and asset.get("anchor_file_path")
        and _is_full_product_anchor(asset)
    ]
    if not candidates:
        return None
    return _best_asset_with_geometric_fallback(candidates)


def _geometric_asset_ranking(asset: dict[str, Any]) -> tuple[float, float, float]:
    """当多模态缺失或质量分平手时，按几何特征排序：商品面积占比 > 清晰度 > 检测置信度。"""

    primary_product = asset.get("primary_product", {})
    candidates = primary_product.get("candidates", [])
    area_ratio = float(candidates[0].get("area_ratio", 0.0)) if candidates else 0.0

    preprocess_results = asset.get("preprocess_results", {})
    sharpness_score = float(preprocess_results.get("sharpness_score", 0.0))

    detection_score = float(candidates[0].get("score", 0.0)) if candidates else 0.0

    return (area_ratio, sharpness_score, detection_score)


def _best_asset_with_geometric_fallback(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """按quality_score选最佳素材，平手时按几何排序防止取上传顺序第一张。"""

    if not candidates:
        return None

    quality_scores = [int(asset.get("quality_score", 0) or 0) for asset in candidates]
    max_quality = max(quality_scores) if quality_scores else 0

    # 检查是否所有候选素材的quality_score都相同（多模态失败或全部平手）
    if len(set(quality_scores)) == 1 or max_quality == 0:
        # 退回几何排序：优先面积占比大、清晰、检测置信度高的
        return max(candidates, key=_geometric_asset_ranking)

    # 正常情况：按quality_score选择
    return max(candidates, key=lambda asset: int(asset.get("quality_score", 0) or 0))


def _is_full_product_anchor(asset: dict[str, Any] | None) -> bool:
    """判断素材能否约束完整商品外观，而不只是局部细节。"""

    if not asset:
        return False
    visual_role = str(asset.get("visual_role", "")).strip()
    if visual_role in {"appearance_anchor", "full_product_anchor"}:
        return True
    roles = _string_list(asset.get("normalized_roles", []))
    capabilities = _safe_dict(asset.get("material_capabilities"))
    return "appearance_anchor_candidate" in roles or bool(capabilities.get("appearance_anchor_candidate"))


def _is_real_anchor_asset(asset: dict[str, Any] | None) -> bool:
    """第一版真实锚点条件：有可渲染文件路径，且不是局部细节图。"""

    if not asset:
        return False
    return bool(str(asset.get("file_path", "")).strip()) and _is_full_product_anchor(asset)


def _shot_requires_full_product_anchor(shot: dict[str, Any]) -> bool:
    """整机展示镜头必须使用完整商品锚点，避免从 Logo 局部凭空补全机身。"""

    narrative_role = str(shot.get("narrative_role", "")).strip().lower()
    if narrative_role in {"feature_demo", "detail_proof", "detail_closeup"}:
        material_strategy = str(shot.get("material_strategy", "")).strip()
        selected_skill = str(shot.get("selected_prompt_skill", "")).strip()
        asset_usage = shot.get("asset_usage") or {}
        visual_role = str(asset_usage.get("visual_role", "")).strip()
        if (
            material_strategy == "detail_reference"
            or selected_skill.startswith("detail_reference")
            or visual_role in {"detail_reference", "logo_detail", "brand_detail"}
        ):
            return False
    return narrative_role in {"product_reveal", "product_hero", "feature_demo", "cta"}



def _merge_structured_requirements_into_identity_card(
    identity_card: dict[str, Any],
    task_data: dict[str, Any],
    structured_requirements: dict[str, Any] | None,
) -> None:
    identity_card = _safe_dict(identity_card)
    if not structured_requirements:
        structured_requirements = task_data.get("structured_requirements")
    structured_requirements = _safe_dict(structured_requirements)

    if not structured_requirements:
        return

    existing_must_preserve = set(identity_card.get("must_preserve", []))

    asset_appearance_facts = []
    appearance_summary = str(identity_card.get("appearance_summary", "")).strip()
    if appearance_summary and appearance_summary != "unknown":
        asset_appearance_facts.append(appearance_summary)
    for component in identity_card.get("key_components", []):
        asset_appearance_facts.append(f"保持{component}")

    for fact in asset_appearance_facts:
        existing_must_preserve.add(fact)

    user_must_preserve = structured_requirements.get("must_preserve", [])
    for item in user_must_preserve:
        existing_must_preserve.add(str(item).strip())

    chat_history = task_data.get("chat_history", "")
    if isinstance(chat_history, list):
        chat_history = " ".join(str(c) for c in chat_history)
    if _mentions_preserve_brand_mark(chat_history):
        existing_must_preserve.add("品牌标识")
        existing_must_preserve.add("商标区域")
        existing_must_preserve.add("Logo 形状和位置")
    if "颜色不能变" in chat_history or "颜色不能改" in chat_history:
        existing_must_preserve.add("主色")

    identity_card["must_preserve"] = list(existing_must_preserve)

    existing_avoid = set(identity_card.get("forbidden_changes", []))
    user_avoid = structured_requirements.get("avoid", [])
    for item in user_avoid:
        existing_avoid.add(str(item).strip())

    product_type = str(identity_card.get("product_type", "")).strip()
    product_type_inferred_avoid = _infer_avoid_from_product_type(product_type)
    for item in product_type_inferred_avoid:
        existing_avoid.add(item)

    identity_card["forbidden_changes"] = list(existing_avoid)

    motion_affordance = identity_card.get("motion_affordance", {})
    if isinstance(motion_affordance, dict):
        product_type_motion_bounds = _infer_motion_boundaries_from_product_type(product_type)
        motion_affordance.update(product_type_motion_bounds)
        identity_card["motion_affordance"] = motion_affordance

    identity_card["appearance_anchor_available"] = bool(
        identity_card.get("reference_asset_ids")
        or any(
            p.get("visual_role") == "appearance_anchor"
            for p in task_data.get("asset_profiles", [])
        )
    )


def _infer_avoid_from_product_type(product_type: str) -> list[str]:
    avoid = []
    electronics = {"手机", "笔记本电脑", "平板", "耳机", "音箱", "相机"}
    food = {"零食", "饮料", "茶叶", "咖啡"}
    clothing = {"服饰", "鞋", "饰品"}

    if product_type in electronics:
        avoid.extend(["改变屏幕显示内容", "改变接口布局", "改变按键形状"])
    elif product_type in food:
        avoid.extend(["改变包装外观", "改变食物颜色和形态"])
    elif product_type in clothing:
        avoid.extend(["改变面料纹理", "改变剪裁轮廓"])
    return avoid


def _mentions_preserve_brand_mark(text: str) -> bool:
    """识别用户对商标、Logo、品牌标识保持不变的口语化要求。"""

    normalized = text.replace(" ", "").lower()
    mark_words = ("商标", "logo", "品牌标识", "标志", "标识")
    preserve_words = ("不能变", "不能改", "不要变", "不要改", "保持", "保留", "不变", "不改")
    return any(mark in normalized for mark in mark_words) and any(word in normalized for word in preserve_words)


def _infer_motion_boundaries_from_product_type(product_type: str) -> dict[str, Any]:
    boundaries: dict[str, Any] = {}
    fragile_types = {"玻璃杯", "陶瓷", "花瓶", "酒杯"}
    electronics = {"手机", "笔记本电脑", "平板", "耳机", "音箱", "相机"}
    if product_type in fragile_types:
        boundaries["forbidden_actions"] = ["快速移动", "碰撞", "翻转"]
        boundaries["max_rotation_degrees"] = 15
    elif product_type in electronics:
        boundaries["forbidden_actions"] = ["不合理翻转", "悬浮", "变形"]
        boundaries["max_rotation_degrees"] = 30
    return boundaries



def build_asset_capability_plan(
    asset_analysis: dict[str, Any],
    product_context: dict[str, Any],
) -> dict[str, Any]:
    """把素材画像转换为可拍/不可拍镜头约束。

    这个模块是生成前的硬约束入口：后续剧本、分镜和素材匹配都应该服从它，
    避免先幻想复杂剧情，再让视频模型凭空补商品、logo 或使用动作。
    """

    assets = asset_analysis.get("assets", [])
    asset_profiles = asset_analysis.get("asset_profiles", [])
    identity_card = _safe_dict(asset_analysis.get("product_identity_card", {}))
    motion_affordance = _safe_dict(identity_card.get("motion_affordance", {}))

    image_assets = [
        a for a in assets
        if a.get("asset_type") == "image" and a.get("is_supported") and a.get("file_path")
    ]
    appearance_assets = [
        a for a in image_assets
        if _is_full_product_anchor(a)
    ]
    detail_assets = [
        a for a in image_assets
        if str(a.get("visual_role", "")).strip() == "detail_reference"
        or "detail_closeup" in a.get("suitable_for", [])
    ]
    scene_assets = [
        a for a in image_assets
        if str(a.get("visual_role", "")).strip() == "scene_context"
    ]

    has_anchor = bool(appearance_assets or identity_card.get("appearance_anchor_available"))
    has_detail = bool(detail_assets or appearance_assets)
    has_scene = bool(scene_assets)

    supported: list[str] = []
    unsupported: list[str] = []
    missing_assets: list[dict[str, str]] = []

    if has_anchor:
        supported.extend(["product_reveal", "product_hero", "static_feature_showcase", "cta_packshot"])
    else:
        unsupported.extend(["product_reveal", "product_hero", "static_feature_showcase", "cta_packshot"])
        missing_assets.append({"needed_for": "product_identity", "suggestion": "补充一张主体完整、logo 清晰、无遮挡的商品主图。"})

    if has_detail:
        supported.append("detail_closeup")
    else:
        missing_assets.append({"needed_for": "detail_closeup", "suggestion": "补充 logo、材质、接口或结构细节图。"})

    if has_scene:
        supported.append("lifestyle_context")
    else:
        # 场景镜头可以 text_to_video，但不能展示真实商品。
        supported.append("generic_problem_scene_without_product")

    # 这些动作在带货商品生成里风险极高，除非用户提供明确使用素材，否则默认禁止。
    unsupported.extend([
        "complex_hand_interaction",
        "open_close_action",
        "free_rotation",
        "flying_product",
        "liquid_pouring",
        "precise_logo_regeneration",
        "new_side_view_from_single_image",
    ])

    allowed_actions = motion_affordance.get("allowed_actions", []) or []
    forbidden_actions = motion_affordance.get("forbidden_actions", []) or []

    # 决定推荐的镜头序列策略
    best_anchor_for_strategy = _find_best_appearance_anchor(asset_analysis) if has_anchor else None
    sequence_strategy = _decide_shot_sequence_strategy(has_anchor, best_anchor_for_strategy, product_context)

    if sequence_strategy == "material_first_expand":
        # 从素材首帧出发扩展
        recommended_structure = [
            "product_reveal",  # 第0镜：从素材首帧建立真实商品
            "product_hero" if has_anchor else "static_feature_showcase",
            "detail_closeup" if has_detail else "usage_context",
            "cta_packshot" if has_anchor else "cta_text_only",
        ]
    else:
        # 旧策略：无商品铺垫开场
        recommended_structure = [
            "generic_problem_scene_without_product",
            "product_reveal" if has_anchor else "cta_text_only",
            "product_hero" if has_anchor else "cta_text_only",
            "detail_closeup" if has_detail else "static_feature_showcase",
            "cta_packshot" if has_anchor else "cta_text_only",
        ]

    return {
        "global_asset_confidence": "high" if has_anchor else ("medium" if image_assets else "low"),
        "appearance_anchor_available": has_anchor,
        "detail_anchor_available": has_detail,
        "scene_asset_available": has_scene,
        "supported_shot_types": _unique_list(supported),
        "unsupported_shot_types": _unique_list(unsupported),
        "recommended_story_structure": recommended_structure,
        "sequence_strategy": sequence_strategy,
        "missing_assets": missing_assets,
        "allowed_actions": allowed_actions,
        "forbidden_actions": _unique_list(forbidden_actions + ["悬浮", "飞入", "大角度翻转", "凭空变形", "logo 重绘"]),
        "planning_rule": "剧情必须服从素材能力；商品镜头没有真实外观锚点时，不允许文生视频生成可识别商品。",
    }


def review_storyboard_shootability(
    storyboard: list[dict[str, Any]],
    product_context: dict[str, Any],
    asset_analysis: dict[str, Any],
) -> dict[str, Any]:
    """生成前可拍性检查：在调用视频 API 前发现高风险镜头。"""

    issues: list[dict[str, Any]] = []
    available_asset_ids = {
        str(a.get("asset_id", "")).strip()
        for a in asset_analysis.get("assets", [])
        if a.get("asset_type") == "image" and a.get("is_supported") and a.get("file_path")
    }
    capability = _safe_dict(product_context.get("asset_capability_plan", {}))
    has_anchor = bool(capability.get("appearance_anchor_available"))
    unsupported_types = set(str(i) for i in capability.get("unsupported_shot_types", []))

    for shot in storyboard:
        shot_index = int(shot.get("shot_index", len(issues) + 1) or 1)
        product_presence = str(shot.get("product_presence", "optional")).strip().lower()
        render_strategy = str(shot.get("render_strategy", "")).strip()
        asset_id = str(shot.get("asset_id", "")).strip()
        role = str(shot.get("narrative_role", "")).strip().lower()
        text = _shot_text_for_risk(shot)

        if product_presence == "required" and not asset_id and not has_anchor:
            issues.append(_shootability_issue(shot_index, "required_product_without_asset", "商品主镜头缺少真实外观锚点，不能让文生视频凭空生成商品。"))
        elif product_presence == "required" and asset_id and asset_id not in available_asset_ids:
            issues.append(_shootability_issue(shot_index, "invalid_asset_id", f"分镜绑定了不存在或不可用的素材：{asset_id}"))

        if product_presence == "required" and render_strategy == "text_to_video":
            issues.append(_shootability_issue(shot_index, "text_to_video_product_risk", "可识别商品镜头不能使用 text_to_video。"))

        if _text_contains_any(text, _HIGH_RISK_MOTION_WORDS):
            issues.append(_shootability_issue(shot_index, "high_risk_motion", "镜头包含翻转、悬浮、飞入、倒液体或复杂开合等高风险动作。"))

        if (
            _text_contains_any(text, _HAND_INTERACTION_WORDS)
            and not _has_usage_asset(asset_analysis)
            and not _is_product_fidelity_v3_allowed_action(shot)
        ):
            issues.append(_shootability_issue(shot_index, "hand_interaction_without_usage_asset", "没有真实手持/使用素材时，不建议生成精细手部交互。"))

        if role in unsupported_types:
            issues.append(_shootability_issue(shot_index, "unsupported_shot_type", f"镜头角色 {role} 不在当前素材能力支持范围内。"))

        if _camera_motion_overloaded(str(shot.get("camera_motion", ""))):
            issues.append(_shootability_issue(shot_index, "camera_overload", "单镜头运镜过载，容易导致主体和 logo 漂移。"))

    return {
        "passed": not issues,
        "issues": issues,
        "issue_count": len(issues),
    }


def repair_storyboard_by_shootability(
    storyboard: list[dict[str, Any]],
    review: dict[str, Any],
    product_context: dict[str, Any],
    asset_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """用确定性规则修复高风险分镜，避免再次调用 LLM 拖慢流程。"""

    issues_by_shot: dict[int, list[dict[str, Any]]] = {}
    for issue in review.get("issues", []):
        issues_by_shot.setdefault(int(issue.get("shot_index", 0) or 0), []).append(issue)

    best_anchor = _best_appearance_anchor_asset(asset_analysis.get("assets", []))
    capability = _safe_dict(product_context.get("asset_capability_plan", {}))
    has_anchor = bool(best_anchor or capability.get("appearance_anchor_available"))

    repaired: list[dict[str, Any]] = []
    for raw_shot in storyboard:
        shot = dict(raw_shot)
        shot_index = int(shot.get("shot_index", len(repaired) + 1) or len(repaired) + 1)
        shot_issues = issues_by_shot.get(shot_index, [])
        issue_types = {str(i.get("type", "")) for i in shot_issues}
        if not shot_issues:
            repaired.append(shot)
            continue

        risk_notes = list(shot.get("risk_notes", []) or [])
        risk_notes.extend(str(i.get("message", "")) for i in shot_issues if i.get("message"))
        shot["risk_notes"] = _unique_list(risk_notes)

        if "invalid_asset_id" in issue_types:
            shot["asset_id"] = ""

        if "text_to_video_product_risk" in issue_types and has_anchor:
            shot["render_strategy"] = "image_to_video"
            shot["identity_strictness"] = "high"
            if best_anchor and not shot.get("asset_id"):
                shot["asset_id"] = best_anchor.get("asset_id", "")

        if "required_product_without_asset" in issue_types and best_anchor:
            shot["asset_id"] = best_anchor.get("asset_id", "")
            shot["render_strategy"] = "image_to_video"
            shot["identity_strictness"] = "high"
        elif "required_product_without_asset" in issue_types and not best_anchor:
            # 没有任何商品图时，不能凭空生成可识别商品，降级为无商品场景 + 文案表达。
            shot["product_presence"] = "forbidden"
            shot["render_strategy"] = "text_to_video"
            shot["asset_id"] = ""
            shot["identity_strictness"] = "low"
            shot["visual_description"] = "真实生活场景中只展示普通无品牌道具，不出现可识别商品主体；通过字幕和口播表达卖点。"
            shot["asset_requirement"] = "缺少商品真实外观锚点，已降级为无商品场景。"

        if issue_types & {"high_risk_motion", "hand_interaction_without_usage_asset", "camera_overload", "unsupported_shot_type"}:
            shot["action"] = _safe_static_action(shot)
            shot["camera_motion"] = "定镜" if str(shot.get("identity_strictness", "")).lower() == "high" else "轻微水平平移"
            shot["visual_description"] = _safe_visual_description(shot)
            forbidden = list(shot.get("forbidden_variation", []) or [])
            forbidden.extend(["商品变形", "logo 漂移", "悬浮", "飞入", "翻转", "复杂手部交互"])
            shot["forbidden_variation"] = _unique_list(forbidden)
            focus = list(shot.get("review_focus", []) or [])
            focus.extend(["商品外观一致性", "物理合理性", "是否移除高风险动作"])
            shot["review_focus"] = _unique_list(focus)

        repaired.append(shot)

    return repaired


def _enforce_storyboard_continuity_groups(storyboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """用规则补齐同商品连续镜头的续写字段，不完全依赖 LLM。"""

    product_roles = {"product_reveal", "feature_demo", "detail_proof", "product_hero", "cta"}
    last_product_group = "product_showcase"
    previous_product_required = False
    result: list[dict[str, Any]] = []
    for raw_shot in storyboard:
        shot = dict(raw_shot)
        role = str(shot.get("narrative_role", "")).strip().lower()
        product_presence = str(shot.get("product_presence", "optional")).strip().lower()
        render_strategy = str(shot.get("render_strategy", "")).strip()
        is_product_shot = product_presence == "required" or (role in product_roles and render_strategy == "image_to_video")
        if is_product_shot:
            shot["continuity_group"] = shot.get("continuity_group") or last_product_group
            if str(shot.get("planner_source", "")).startswith("product_fidelity_v3"):
                shot["transition_type"] = "hard_cut"
                shot["continuity_group"] = ""
                shot["anchor_last_frame"] = False
                previous_product_required = True
                result.append(shot)
                continue
            if previous_product_required and shot.get("transition_type") not in {"hard_cut", "crossfade"}:
                shot["transition_type"] = "continue_from_previous"
            elif previous_product_required and shot.get("transition_type") == "hard_cut":
                shot["transition_type"] = "continue_from_previous"
            if previous_product_required:
                shot["anchor_last_frame"] = bool(shot.get("anchor_last_frame", True))
            previous_product_required = True
        else:
            previous_product_required = False
        result.append(shot)
    return result


_HIGH_RISK_MOTION_WORDS = (
    "悬浮", "飞入", "飞出", "凭空", "变形", "翻转", "旋转展示", "大幅旋转", "环绕展示",
    "倒入", "倒出", "飞溅", "切开", "撕开", "拆开", "打开包装", "自动打开", "折叠", "展开",
)

_HAND_INTERACTION_WORDS = ("拿起", "握住", "按下", "打开", "合上", "拆开", "撕开", "插入", "拔出", "佩戴", "戴上")


def _shot_text_for_risk(shot: dict[str, Any]) -> str:
    return " ".join(
        str(shot.get(key, ""))
        for key in ("narrative_role", "scene_goal", "purpose", "visual_description", "action", "camera_motion", "asset_requirement")
    )


def _shootability_issue(shot_index: int, issue_type: str, message: str) -> dict[str, Any]:
    return {"shot_index": shot_index, "type": issue_type, "message": message}


def _text_contains_any(text: str, words: tuple[str, ...]) -> bool:
    normalized = str(text).replace(" ", "")
    return any(_contains_positive_motion_word(normalized, word) for word in words)


def _contains_positive_motion_word(text: str, word: str) -> bool:
    start = 0
    negation_markers = ("不", "禁", "禁止", "不能", "不得", "避免", "严禁", "无")
    while True:
        index = text.find(word, start)
        if index < 0:
            return False
        prefix = text[max(0, index - 4) : index]
        if not any(marker in prefix for marker in negation_markers):
            return True
        start = index + len(word)


def _has_usage_asset(asset_analysis: dict[str, Any]) -> bool:
    for asset in asset_analysis.get("assets", []):
        suitable = asset.get("suitable_for", []) or []
        visual_role = str(asset.get("visual_role", ""))
        background_type = str(asset.get("background_type", ""))
        if any(key in suitable for key in ("scene_context", "usage_scene", "lifestyle_result")):
            return True
        if visual_role == "scene_context" or "场景" in background_type:
            return True
    return False


def _is_product_fidelity_v3_allowed_action(shot: dict[str, Any]) -> bool:
    source = str(shot.get("planner_source", ""))
    if not source.startswith("product_fidelity_v3"):
        return False
    action = str(shot.get("action", ""))
    material_strategy = str(shot.get("material_strategy", "")).strip()
    selected_skill = str(shot.get("selected_prompt_skill", "")).strip()
    if material_strategy == "source_scene_extension" or selected_skill.startswith("source_scene_extension."):
        disallowed = ("拿起", "拿离", "提起", "打开", "开合", "旋转", "翻转", "走路", "嘴边", "喝水", "倒")
        if any(word in action for word in disallowed):
            motion_affordance = _safe_dict((shot.get("product_identity_card") or {}).get("motion_affordance"))
            allowed_actions = " ".join(_string_list(motion_affordance.get("allowed_actions")))
            return any(word in allowed_actions for word in ("拿起", "拿离", "提起")) and not any(
                word in action for word in ("打开", "开合", "旋转", "翻转", "走路", "嘴边", "喝水", "倒")
            )
        return True
    return bool(action.strip())


def _camera_motion_overloaded(camera_motion: str) -> bool:
    # 单独的高风险运镜词（任意一个即触发）
    high_risk_single = ("环绕", "旋转", "翻转", "推近", "缓慢推近", "推入")
    if any(_contains_positive_motion_word(camera_motion.replace(" ", ""), word) for word in high_risk_single):
        return True
    # 多词叠加也触发
    motion_words = ("推", "拉", "摇", "移", "跟", "升", "降")
    return sum(1 for word in motion_words if word in camera_motion) >= 2


def _safe_static_action(shot: dict[str, Any]) -> str:
    product_presence = str(shot.get("product_presence", "optional")).strip().lower()
    if product_presence == "forbidden":
        return "场景保持稳定，只允许人物或普通无品牌道具出现自然轻微动作。"
    return "商品保持稳定，不改变结构、颜色和标识；只允许轻微自然光影变化。"


def _safe_visual_description(shot: dict[str, Any]) -> str:
    product_presence = str(shot.get("product_presence", "optional")).strip().lower()
    subtitle = str(shot.get("subtitle", "")).strip()
    if product_presence == "forbidden":
        return f"真实生活场景，画面不出现可识别商品主体、logo 或品牌文字；用字幕「{subtitle}」表达当前卖点。"
    return (
        "商品位于画面中央或主体区域，保持上传素材中的真实外观、颜色、材质、结构和标识不变；"
        f"通过稳定构图、柔和光线和字幕「{subtitle}」表达卖点，不做复杂手部操作。"
    )


def _unique_list(items: list[Any]) -> list[Any]:
    seen = set()
    result = []
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True) if isinstance(item, dict) else str(item)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result

def _director_context_for_script(
    product_context: dict[str, Any],
    asset_analysis: dict[str, Any],
) -> dict[str, Any]:
    """为剧本生成提供轻量创意上下文，不需要完整的导演决策。"""

    identity_card = product_context.get("product_identity_card", {})
    asset_profiles = asset_analysis.get("asset_profiles", [])
    return {
        "product_type": product_context.get("product_type", ""),
        "appearance_summary": identity_card.get("appearance_summary", ""),
        "selling_points": product_context.get("selling_points", []),
        "must_preserve": identity_card.get("must_preserve", []),
        "forbidden_changes": identity_card.get("forbidden_changes", []),
        "asset_capability_plan": product_context.get("asset_capability_plan", {}),
        "available_asset_roles": [
            {
                "asset_id": p.get("asset_id", ""),
                "suitable_for": p.get("suitable_for", []),
                "visual_role": p.get("visual_role", ""),
            }
            for p in asset_profiles
        ],
    }


def plan_director_storyboard(
    product_context: dict[str, Any],
    script_plan: dict[str, Any],
    asset_analysis: dict[str, Any],
    previous_issues: list[str] | None = None,
) -> list[dict[str, Any]]:
    """导演+分镜合并节点：基于剧本和素材，一次 LLM 调用产出带素材绑定的完整分镜。

    之前 plan_director_decision + build_storyboard 分两次调用：
    - 导演选了策略和素材建议，但是自然语言文本
    - 分镜 LLM 可能忽略导演的素材建议
    - 加上 plan_script 一共 3 次文本 LLM 调用

    现在合并为 1 次调用，asset_id 作为结构化的硬输出直接落在每镜上。
    """

    print("[video_generation_workflow] 开始导演+分镜联合生成。", flush=True)
    duration = max(3, int(product_context.get("duration_seconds", 15)))
    identity_card = product_context.get("product_identity_card", {})
    assets = asset_analysis.get("assets", [])
    asset_profiles = asset_analysis.get("asset_profiles", [])

    prompt = {
        "task": f"你是带货视频导演。根据剧本和素材，直接输出 {STORYBOARD_MIN_SHOTS}-{STORYBOARD_MAX_SHOTS} 个完整分镜，每个分镜明确绑定素材和渲染策略。",
        "thinking_steps": [
            "1. 理解剧本的叙事弧线和每个 beat 的表达目标",
            "2. 严格继承 beat 中的 scene_before、action、scene_after、shot_type、camera_movement、subject_position、dialogue、scene_elements、physical_constraints 和 asset_requirements。导演可以取舍和细化，但不能重新编造另一套故事",
            "3. 先查看 asset_capability_plan，只使用 supported_shot_types 支持的镜头表达；unsupported_shot_types 中的动作必须用字幕/旁白替代",
            "4. 从 hook、problem、context、product_reveal、feature_demo、detail_proof、lifestyle_result、cta 中选择适合当前商品的 3-7 个镜头，不要求固定模板数量",
            "5. 展示真实商品外观、logo、材质或结构的镜头必须使用 image_to_video 并填写 asset_id",
            "5. text_to_video 只用于不展示可识别商品主体的环境铺垫镜头，不得突出品牌标识；只要商品可被识别就必须改用 image_to_video",
            "6. 填写审核字段：product_presence、identity_strictness、review_focus、completion_criteria",
            "7. 至少安排一个独立剧情场景，再安排一个 product_reveal 商品揭示镜。剧情场景不展示可识别商品；product_reveal 有真实素材时必须直接绑定 appearance_anchor，并使用 crossfade 从前一场景切换",
            "8. 字幕、CTA 文案、按钮和购物车图标只写入 subtitle 或 voiceover。visual_description、action 和 final_state 不得要求画面内出现文字、图标、按钮或 UI",
        ],
        "script_plan": script_plan,
        "product_info": _product_context_for_llm(product_context),
        "product_identity_card": identity_card,
        "asset_capability_plan": product_context.get("asset_capability_plan", {}),
        "available_assets": [
            {
                "asset_id": p.get("asset_id", ""),
                "filename": p.get("filename", ""),
                "suitable_for": p.get("suitable_for", []),
                "visual_role": p.get("visual_role", ""),
                "quality_score": p.get("quality_score", 80),
                "identity_contribution": p.get("identity_contribution", []),
            }
            for p in asset_profiles
        ],
        "duration_limit_seconds": duration,
        "shot_count_range": {
            "min": STORYBOARD_MIN_SHOTS,
            "max": STORYBOARD_MAX_SHOTS,
            "rule": f"分镜数量必须落在 {STORYBOARD_MIN_SHOTS}-{STORYBOARD_MAX_SHOTS} 范围内。",
        },
        "render_strategies": {
            "image_to_video": "展示商品外观、logo、材质或结构时使用。必须填写 asset_id，素材作为真实外观锚点。",
            "text_to_video": "仅用于不展示可识别商品主体的场景铺垫镜头。asset_id 留空，不得出现或突出品牌标识。",
        },
        "physical_constraints": (
            "所有运动必须可物理实现。logo、材质、结构特写使用定镜，不推近、不拉远、不旋转、不环绕。"
            "普通商品展示最多允许幅度不超过画面 10% 的轻微水平平移。"
            "场景铺垫镜头可使用定镜或轻微平移，不得描述任何超现实变形、折叠、翻转或不合理角度变化。"
        ),
        "hard_rules": {
            "asset_binding": "每个镜头如果有对应的上传素材图，必须填写 asset_id；有 asset_id 的镜头 render_strategy 必须是 image_to_video",
            "recognizable_product_binding": "只要镜头中出现可识别商品主体、logo、材质或结构，就必须使用 image_to_video 并绑定 asset_id。text_to_video 只能生成环境铺垫。",
            "brand_identity": f"商品身份卡中的 must_preserve 必须作为 hard constraint 传递到每个 product_presence=required 的镜头：{identity_card.get('must_preserve', [])}",
            "forbidden": f"商品身份卡中的 forbidden_changes 不能出现在任何镜头描述中：{identity_card.get('forbidden_changes', [])}",
            "visual_effect_ban": "不要描述烟雾、雾气、蒸汽、尘埃、光束、粒子飘浮等抽象氛围特效。只描述桌面、商品、书本、灯光等可见实物。",
            "story_scene_required": "不能把所有镜头都写成商品图片轻微运动。至少一个 hook/problem/context/lifestyle_result 镜头使用 text_to_video 展示具体场景，并禁止突出可识别商品。",
            "product_reveal_bridge": "在第一个商品事实镜之前安排 product_reveal。有 appearance_anchor 时必须直接绑定真实素材 asset_id，render_strategy 填 image_to_video，transition_type 填 crossfade，continuity_mode 留空。只有完全没有可用真实商品锚点时，才允许 continuity_mode 填 shared_scene_bridge。",
            "local_overlay_only": "字幕、CTA 文案、按钮和图标统一由本地后处理叠加。visual_description、action、initial_state 和 final_state 禁止要求视频模型绘制文字、图标、按钮或 UI。",
            "continuity_design": "同一空间内需要连续动作时，为镜头填写相同 continuity_group，后续镜头 transition_type 填 continue_from_previous。跨空间换场默认填 hard_cut；只有刻意表达时间或氛围变化时才填 crossfade。",
            "product_anchor_reset": "同一商品连续展示时，也可以使用相同 continuity_group 和 continue_from_previous。后续商品镜填写 anchor_last_frame=true，使模型在镜头末尾回到真实上传素材，减少累计漂移。",
            "asset_capability": "禁止生成 asset_capability_plan.unsupported_shot_types 中的镜头。如果缺少真实素材，不要让 text_to_video 生成可识别商品。",
            "scene_fidelity": "visual_description、subject_appearance、acting_direction 必须直接继承 script_plan.beats 中对应 beat 的 scene_description、subject_appearance、acting_direction。导演可以细化，但不能发明新场景替换剧本中的场景。",
        },
        "output_format": {
            "shot_index": "分镜序号",
            "duration_seconds": "时长（秒）",
            "narrative_role": "hook / problem / context / product_reveal / feature_demo / detail_proof / lifestyle_result / cta",
            "scene_goal": "这个镜头要完成的表达目标",
            "purpose": "这个镜头存在的原因，必须与 rich_story_text 中的叙事推进对应",
            "initial_state": "镜头开始时可见的空间、主体和动作状态",
            "action": "镜头中发生的单一核心动作或状态变化，必须物理合理",
            "final_state": "镜头结束时的可见状态，供下一镜判断如何承接",
            "shot_type": "特写 / 近景 / 中景 / 全景",
            "camera_motion": "定镜 / 轻微水平平移。logo、材质、结构特写必须使用定镜",
            "subject_appearance": "镜头中主体的外观描述，人物穿什么衣服、姿态；商品颜色材质结构；连续镜头必须与上一镜保持一致",
            "subject_position": "主体和关键道具在画面中的具体位置关系（如：人物位于画面左侧三分之一，商品在右侧桌面）",
            "acting_direction": "人物或商品的具体动作指导，要可物理实现的精确描述（如：用户右手从包侧袋取出水杯，拧开杯盖约90度）",
            "scene_elements": ["需要跨镜头持续跟踪的具体环境元素和道具"],
            "visual_description": "画面描述——精确的物理动作 + 构图 + 光线",
            "subtitle": "叠加字幕文案，用户可见的完整短句，建议 12-24 个中文字符；不要写内部标签或促销套话",
            "voiceover": "口播文案；可比 subtitle 更完整，但不要写镜头说明",
            "asset_id": "绑定的素材 ID，有对应素材时必须填写；没有则留空",
            "asset_requirement": "说明该镜头需要哪类素材，以及为什么需要或不需要真实商品素材",
            "render_strategy": "image_to_video（有素材）或 text_to_video（无素材）",
            "continuity_mode": "默认留空。只有 product_reveal 完全没有可用真实商品锚点时才填 shared_scene_bridge",
            "continuity_group": "同一空间内需要连续生成的镜头填写相同组名，例如 desk_story；不需要续写则留空",
            "transition_type": "hard_cut / continue_from_previous / crossfade。默认 hard_cut，不能把所有镜头都写成 crossfade",
            "anchor_last_frame": "布尔值。商品连续展示镜头使用 continue_from_previous 时填 true，让真实素材作为目标尾帧",
            "product_presence": "required / optional / forbidden",
            "identity_strictness": "high / medium / low",
            "allowed_variation": ["允许变化的内容"],
            "forbidden_variation": ["禁止变化的内容"],
            "review_focus": ["审查重点"],
            "completion_criteria": ["完成标准"],
        },
        "instruction": "只返回 JSON 数组，不要返回解释文字。",
        "previous_issues": previous_issues or [],
    }
    llm_result = _call_text_llm(prompt, purpose="director_storyboard", temperature=0.85)

    if llm_result["ok"]:
        parsed = _extract_json_from_text(llm_result["content"])
        storyboard = _normalize_storyboard(parsed)
        if storyboard:
            print(
                "[video_generation_workflow] 导演+分镜完成："
                f"shot_count={len(storyboard)}",
                flush=True,
            )
            return _ensure_storyboard_continuity(storyboard)
        print("[video_generation_workflow] 导演+分镜失败：LLM 返回内容无法解析。", flush=True)
        return []

    # LLM 不可用时走兜底，但兜底分镜必须保留素材绑定。
    return _fallback_director_storyboard(product_context, script_plan, asset_analysis)


def _fallback_director_storyboard(
    product_context: dict[str, Any],
    script_plan: dict[str, Any],
    asset_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """导演+分镜 LLM 失败时的规则兜底，保留最小剧情层次和真实素材绑定。"""

    duration = max(8, int(product_context.get("duration_seconds", 15)))
    identity_card = product_context.get("product_identity_card", {})
    fallback_assets = [
        asset
        for asset in asset_analysis.get("assets", [])
        if asset.get("asset_type") == "image" and asset.get("is_supported")
    ]
    reveal_asset = _best_appearance_anchor_asset(fallback_assets) or (fallback_assets[0] if fallback_assets else None)
    reveal_asset_id = str((reveal_asset or {}).get("asset_id", "")).strip()
    has_reveal_asset = bool(reveal_asset_id)
    body = [str(item).strip() for item in script_plan.get("body", []) if str(item).strip()]
    caption_context = dict(product_context)
    if body and not caption_context.get("selling_points"):
        caption_context["selling_points"] = body
    hook = _safe_user_caption(
        str(script_plan.get("hook", "")).strip(),
        fallback=_fallback_public_caption(caption_context, "hook", scene_goal="使用场景看清"),
        max_chars=14,
    )
    reveal_caption = _fallback_public_caption(caption_context, "product_reveal", scene_goal="外观细节看清")
    cta = _safe_user_caption(
        str(script_plan.get("cta", "")).strip(),
        fallback=_fallback_public_caption(caption_context, "cta", scene_goal="卖点结果看得见"),
        max_chars=DEFAULT_SUBTITLE_MAX_CHARS,
    )
    first_point = body[0] if body else "核心卖点"
    second_point = body[1] if len(body) > 1 else "真实细节"
    first_point_caption = _safe_user_caption(
        first_point,
        fallback=_fallback_public_caption(caption_context, "feature_demo", scene_goal="核心卖点看清"),
        max_chars=DEFAULT_SUBTITLE_MAX_CHARS,
    )
    second_point_caption = _safe_user_caption(
        second_point,
        fallback=_fallback_public_caption(caption_context, "detail_proof", scene_goal="细节看得见"),
        max_chars=DEFAULT_SUBTITLE_MAX_CHARS,
    )

    # 按 15 秒常用节奏分配，再把差值加到核心展示镜，兼容 8-15 秒任务。
    durations = [2, 2, 4, 4, 3]
    duration_delta = duration - sum(durations)
    durations[2] = max(1, durations[2] + duration_delta)

    storyboard = [
        {
            "shot_index": 1,
            "narrative_role": "hook",
            "scene_goal": "用具体生活场景建立与商品卖点相关的使用情境。",
            "initial_state": "真实生活场景保持自然状态。",
            "action": "通过环境和人物动作展示使用情境或目标状态，不出现可识别商品主体。",
            "final_state": "画面停留在与商品价值相关的场景关系上。",
            "duration_seconds": durations[0],
            "subtitle": hook,
            "voiceover": hook,
            "camera_motion": "轻微水平平移",
            "visual_description": "真实生活场景，使用环境和动作表达商品价值相关情境，不展示可识别商品、Logo 或品牌标识。",
            "render_strategy": "text_to_video",
            "continuity_group": "story_scene",
            "transition_type": "hard_cut",
            "product_presence": "forbidden",
            "identity_strictness": "low",
            "asset_requirement": "无需商品素材，生成独立剧情场景。",
            "forbidden_variation": [],
            "review_focus": ["剧情问题是否清楚"],
            "completion_criteria": ["不出现可识别商品主体"],
            "product_identity_constraints": [],
        },
        {
            "shot_index": 2,
            "narrative_role": "product_reveal",
            "continuity_mode": "" if has_reveal_asset else "shared_scene_bridge",
            "scene_goal": "从剧情场景自然切换到真实商品。",
            "initial_state": "统一棚拍背景保持留白。",
            "action": "真实商品锚点在同一背景上平稳淡入。",
            "final_state": "商品完整出现，准备进入卖点展示。",
            "duration_seconds": durations[1],
            "subtitle": reveal_caption,
            "voiceover": reveal_caption,
            "camera_motion": "定镜",
            "visual_description": "统一墙面和桌面背景中，真实商品锚点逐渐出现。",
            "render_strategy": "image_to_video",
            "transition_type": "crossfade",
            "product_presence": "required" if has_reveal_asset else "optional",
            "identity_strictness": "medium",
            "asset_id": reveal_asset_id,
            "asset_requirement": "优先直接使用真实商品外观锚点；没有可用锚点时才使用共享场景底图。",
            "forbidden_variation": ["不改变商品主体外观"],
            "review_focus": ["商品揭示是否自然"],
            "completion_criteria": ["商品从留白背景中完整淡入"],
            "product_identity_constraints": identity_card.get("must_preserve", []),
        },
        {
            "shot_index": 3,
            "narrative_role": "feature_demo",
            "scene_goal": f"用真实商品画面证明卖点：{first_point_caption}",
            "initial_state": "真实商品完整出现在统一背景中。",
            "action": "商品保持结构稳定，镜头只做轻微水平平移。",
            "final_state": "核心卖点得到清晰展示。",
            "duration_seconds": durations[2],
            "subtitle": first_point_caption,
            "voiceover": first_point_caption,
            "camera_motion": "轻微水平平移",
            "visual_description": "使用上传素材中的同一件真实商品，保持外观和结构稳定，展示核心卖点。",
            "render_strategy": "image_to_video",
            "continuity_group": "product_showcase",
            "transition_type": "hard_cut",
            "product_presence": "required",
            "identity_strictness": "high",
            "asset_requirement": "商品完整外观锚点。",
            "forbidden_variation": identity_card.get("must_preserve", []),
            "review_focus": ["商品外观一致性", "核心卖点是否可见"],
            "completion_criteria": ["商品主体完整清晰"],
            "product_identity_constraints": identity_card.get("must_preserve", []),
        },
        {
            "shot_index": 4,
            "narrative_role": "detail_proof",
            "scene_goal": f"用真实细节增强可信度：{second_point_caption}",
            "initial_state": "镜头保持商品主体稳定。",
            "action": "固定构图，利用光线变化展示材质或关键细节。",
            "final_state": "细节清晰停留在画面中。",
            "duration_seconds": durations[3],
            "subtitle": second_point_caption,
            "voiceover": second_point_caption,
            "camera_motion": "定镜",
            "visual_description": "绑定真实商品素材，用稳定构图展示材质、结构或品牌识别区域。",
            "render_strategy": "image_to_video",
            "continuity_group": "product_showcase",
            "transition_type": "continue_from_previous",
            "anchor_last_frame": True,
            "product_presence": "required",
            "identity_strictness": "high",
            "asset_requirement": "商品细节参考图或外观锚点。",
            "forbidden_variation": identity_card.get("must_preserve", []),
            "review_focus": ["细节一致性", "结构稳定性"],
            "completion_criteria": ["细节清晰可见"],
            "product_identity_constraints": identity_card.get("must_preserve", []),
        },
        {
            "shot_index": 5,
            "narrative_role": "cta",
            "scene_goal": "回到商品完整画面并完成行动引导。",
            "initial_state": "商品保持完整展示。",
            "action": "镜头节奏放缓，保留字幕区域。",
            "final_state": "商品稳定定格并结束。",
            "duration_seconds": durations[4],
            "subtitle": cta,
            "voiceover": cta,
            "camera_motion": "定镜",
            "visual_description": "使用上传素材中的同一件真实商品，保持稳定构图并预留字幕空间。",
            "render_strategy": "image_to_video",
            "continuity_group": "product_showcase",
            "transition_type": "continue_from_previous",
            "anchor_last_frame": True,
            "product_presence": "required",
            "identity_strictness": "high",
            "asset_requirement": "商品完整外观锚点。",
            "forbidden_variation": identity_card.get("must_preserve", []),
            "review_focus": ["商品一致性", "收尾完整性"],
            "completion_criteria": ["商品稳定定格", "CTA 清晰"],
            "product_identity_constraints": identity_card.get("must_preserve", []),
        },
    ]
    return _ensure_storyboard_continuity(_bind_fallback_assets(storyboard, asset_analysis))


def _bind_fallback_assets(
    storyboard: list[dict[str, Any]],
    asset_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """为降级分镜按角色分配素材，确保素材绑定在降级路径中不丢失。

    分配规则：
    - hook / cta → 优先选适合 hook_opener 或 cta_closer 的素材
    - feature_demo / detail_proof → 优先选适合 product_showcase 或 detail_closeup 的素材
    - 找不到匹配角色时回退到 index 轮转分配
    """

    image_assets = [
        a for a in asset_analysis.get("assets", [])
        if a.get("asset_type") == "image" and a.get("is_supported") and a.get("file_path")
    ]
    if not image_assets:
        return storyboard

    asset_profiles = asset_analysis.get("asset_profiles", [])
    profile_map: dict[str, dict[str, Any]] = {p.get("asset_id", ""): p for p in asset_profiles}

    # 按角色偏好对素材排序
    hook_cta_assets = []
    showcase_assets = []
    other_assets = []
    for a in image_assets:
        aid = a.get("asset_id", "")
        profile = profile_map.get(aid, {})
        suitable = profile.get("suitable_for", [])
        if any(r in suitable for r in ("hook_opener", "cta_closer")):
            hook_cta_assets.append(a)
        elif any(r in suitable for r in ("product_showcase", "detail_closeup", "product_reveal", "feature_demo", "detail_proof")):
            showcase_assets.append(a)
        else:
            other_assets.append(a)

    hook_cta_index = 0
    showcase_index = 0
    other_index = 0

    for shot in storyboard:
        # 已有 asset_id 的不覆盖
        if shot.get("asset_id"):
            continue
        if shot.get("render_strategy") != "image_to_video":
            continue
        # 商品揭示桥接镜由系统绑定统一场景底图，不直接绑定某张用户素材。
        if shot.get("narrative_role") == "product_reveal":
            continue

        role = shot.get("narrative_role", "")
        if role in ("hook", "cta"):
            pool = hook_cta_assets or image_assets
            idx = hook_cta_index % len(pool)
            hook_cta_index += 1
        elif role in ("feature_demo", "detail_proof"):
            pool = showcase_assets or image_assets
            idx = showcase_index % len(pool)
            showcase_index += 1
        else:
            pool = image_assets
            idx = other_index % len(pool)
            other_index += 1

        matched = pool[idx]
        shot["asset_id"] = matched.get("asset_id", "")

    return storyboard



def _guess_framework_from_storyboard(storyboard: list[dict[str, Any]]) -> str:
    """从分镜角色分布推断使用的叙事框架。"""
    roles = [s.get("narrative_role", "") for s in storyboard]
    if "hook" in roles and "cta" in roles:
        if "detail_proof" in roles:
            return "问题解决"
        if "feature_demo" in roles:
            return "开箱递进"
        return "促销直给"
    return "场景沉浸"


def _guess_camera_style_from_storyboard(storyboard: list[dict[str, Any]]) -> str:
    """从分镜运镜方式推断主运镜风格。"""
    motions = [s.get("camera_motion", "") for s in storyboard]
    push_count = sum(1 for m in motions if "推" in m)
    still_count = sum(1 for m in motions if m == "定镜")
    if push_count >= still_count:
        return "推近"
    return "固定镜头"


def _guess_pacing_from_storyboard(storyboard: list[dict[str, Any]]) -> str:
    """从分镜时长推断节奏风格。"""
    durations = [s.get("duration_seconds", 0) for s in storyboard]
    if not durations:
        return "匀速渐进"
    avg = sum(durations) / len(durations)
    if avg <= 3:
        return "快切全片"
    elif avg >= 7:
        return "慢推全片"
    return "匀速渐进"


def build_product_context(
    task_data: dict[str, Any],
    asset_analysis: dict[str, Any],
) -> dict[str, Any]:
    """整理商品上下文，让后面的剧本和分镜步骤有稳定输入。

    注意：此函数为纯计算，不调用 LLM。audience / creative_goal 使用基于
    商品信息的规则推断，后续由 director_decision 在创意层面进一步细化。
    """

    print("[video_generation_workflow] 开始整理商品上下文。", flush=True)
    structured_requirements = _safe_dict(task_data.get("structured_requirements", {}))
    # 基于商品类型和平台推断受众描述。表单字段优先，结构化摘要作为兜底。
    product_type = str(
        task_data.get("product_type")
        or structured_requirements.get("product_type")
        or ""
    ).strip()
    target_audience_raw = str(
        task_data.get("target_audience")
        or structured_requirements.get("target_audience")
        or ""
    ).strip()
    usage_scene = str(
        task_data.get("usage_scene")
        or structured_requirements.get("usage_scene")
        or ""
    ).strip()
    if target_audience_raw:
        audience = target_audience_raw
    elif product_type:
        audience = f"对{product_type}感兴趣的潜在买家"
    else:
        audience = "对商品外观、便携性和实用性敏感的潜在买家"

    selling_points = _selling_point_phrases(
        task_data.get("selling_points") or structured_requirements.get("selling_point_priority") or []
    )
    title = str(task_data.get("title", ""))
    if selling_points:
        creative_goal = f"用短时间展示{title}的核心卖点「{selling_points[0]}」，让用户产生购买冲动。"
    else:
        creative_goal = "用短时间清楚展示商品卖点，并引导用户点击购买。"

    product_identity_card = _safe_dict(asset_analysis.get("product_identity_card", {}))
    visual_style_bible = _build_visual_style_bible(task_data)
    context = {
        "product_title": title,
        "target_platform": task_data.get("target_platform", "tiktok"),
        "duration_seconds": int(task_data.get("duration_seconds", 15)),
        "style": task_data.get("style", ""),
        "custom_style_prompt": task_data.get("custom_style_prompt", ""),
        "product_type": product_type,
        "target_audience": target_audience_raw,
        "usage_scene": usage_scene,
        "selling_points": selling_points,
        "audience": audience,
        "creative_goal": creative_goal,
        "asset_summary": asset_analysis.get("semantic_summary", ""),
        "asset_profiles": asset_analysis.get("asset_profiles", []),
        "asset_metadata": _asset_metadata_for_llm(asset_analysis.get("assets", [])),
        "reference_image_paths": _image_paths_from_asset_analysis(asset_analysis),
        "product_identity_card": product_identity_card,
        "motion_affordance": _safe_dict(product_identity_card.get("motion_affordance", {})),
        "visual_style_bible": visual_style_bible,
        "structured_requirements": structured_requirements,
        "input_confidence": task_data.get("input_confidence", "medium"),
        "llm_enabled": False,
        "llm_notes": "",
    }
    print("[video_generation_workflow] 商品上下文整理完成。", flush=True)
    return context


def _build_visual_style_bible(task_data: dict[str, Any]) -> dict[str, str]:
    """把表单风格期望整理为整片共享约束，避免每个分镜各自生成一套画风。"""

    custom_style = str(task_data.get("custom_style_prompt", "")).strip()
    style = str(task_data.get("style", "")).strip()
    style_summary = _style_prompt_summary(custom_style, style)
    lighting = "柔和自然光，主体照明稳定"
    background = "背景克制干净，避免复杂装饰抢占商品注意力"
    if style_summary:
        lighting = f"{lighting}；风格：{style_summary}"
        background = f"{background}；风格：{style_summary}"
    return {
        "realism": "真实写实的商业短视频，不使用夸张抽象特效",
        "lighting": lighting,
        "color_temperature": "中性偏暖色温，镜头切换时保持一致",
        "background_complexity": background,
        "camera_language": "稳定镜头，运动幅度小，身份敏感镜头避免翻转、环绕和大角度变化",
        "user_style": style_summary or style or "清晰直接的商品展示风格",
        "style_summary": style_summary or style or "清晰直接的商品展示风格",
    }


def _style_prompt_summary(custom_style_prompt: str, style: str = "") -> str:
    """把模板长 prompt 压缩成短风格摘要，避免污染最终视频 prompt。"""

    text = str(custom_style_prompt or "").strip()
    template_match = re.search(r"【模板：([^】]+)】", text)
    summary_parts: list[str] = []
    if template_match:
        summary_parts.append(template_match.group(1).strip())
    elif style:
        summary_parts.append(str(style).strip())
    user_match = re.search(r"用户补充：(.+)$", text, flags=re.S)
    if user_match:
        user_text = _clean_short_sentence(user_match.group(1), max_chars=36)
        if user_text:
            summary_parts.append(f"用户补充：{user_text}")
    elif text and not template_match:
        user_text = _clean_short_sentence(text, max_chars=36)
        if user_text and user_text not in summary_parts:
            summary_parts.append(user_text)
    return "；".join(part for part in summary_parts if part)


# Path B keeps product-fidelity V3 when a real product anchor exists. Without an anchor it returns to the free director planner so legacy templates cannot bias shot decisions.

def _decide_shot_sequence_strategy(
    has_real_anchor: bool,
    best_anchor: dict[str, Any] | None,
    product_context: dict[str, Any],
) -> str:
    """决定镜头序列策略：优先从素材首帧出发扩展，无锚点时才用自由场景起手。

    Returns:
        "material_first_expand": 存在可用商品锚点时，第0镜从素材首帧建立真实商品，向外扩展
        "free_hook_then_product": 无任何商品锚点时，第0镜无商品铺垫，后续镜补充商品
    """

    if not has_real_anchor or not best_anchor:
        return "free_hook_then_product"

    # 存在可用整机锚点时，默认从素材首帧出发
    return "material_first_expand"


def _plan_product_fidelity_v3_storyboard(
    product_context: dict[str, Any],
    asset_analysis: dict[str, Any],
    best_anchor: dict[str, Any],
) -> list[dict[str, Any]]:
    """商品保真带货 V3：剧情和商品锚定分开，所有镜头使用 Seedance 原生 5 秒。"""

    identity_card = product_context.get("product_identity_card", {})
    product_type = str(identity_card.get("product_type") or product_context.get("product_type") or "商品").strip()
    selling_points = _selling_point_phrases(product_context.get("selling_points", []))
    first_point = selling_points[0] if selling_points else _default_selling_point(product_type)
    second_point = selling_points[1] if len(selling_points) > 1 else first_point
    asset_id = str(best_anchor.get("asset_id", "")).strip()
    source = _product_fidelity_template_source(product_type)
    actions = _product_fidelity_action_templates(product_type)
    feature_action = actions["feature"]
    scene_action = actions["scene"]
    product_action = actions["hook"]
    profiles_by_id = _asset_profile_map(asset_analysis)
    image_assets = [
        asset
        for asset in asset_analysis.get("assets", [])
        if isinstance(asset, dict) and asset.get("asset_type") == "image" and asset.get("is_supported")
    ]
    assets_by_id = {str(asset.get("asset_id", "")).strip(): asset for asset in image_assets}
    allocation = _allocate_assets_to_shot_roles(asset_analysis.get("asset_profiles", []), image_assets)
    reveal_asset = _v3_select_role_asset(allocation, assets_by_id, role="appearance_anchor", fallback=best_anchor)
    reveal_asset = _merge_asset_profile(reveal_asset, profiles_by_id)
    detail_asset = _v3_select_role_asset(allocation, assets_by_id, role="detail_reference", fallback=reveal_asset)
    detail_asset = _merge_asset_profile(detail_asset, profiles_by_id)
    has_real_extension_anchor = _is_real_anchor_asset(reveal_asset)
    reveal_asset_id = str((reveal_asset if has_real_extension_anchor else best_anchor).get("asset_id", asset_id)).strip()
    feature_asset = detail_asset if detail_asset and str(detail_asset.get("asset_id", "")) != reveal_asset_id else reveal_asset
    feature_asset_id = str((feature_asset or {}).get("asset_id", reveal_asset_id)).strip()
    proof_anchor = reveal_asset if isinstance(reveal_asset, dict) and reveal_asset else best_anchor
    appearance_for_plan = _commerce_appearance_text(identity_card, product_type)
    scenario = _commerce_scenario_from_context(product_context, identity_card, product_type)
    risk_level = _commerce_reconstruction_risk(identity_card, proof_anchor)
    proof_plan = _commerce_expression_plan(
        product_context,
        asset_analysis,
        identity_card,
        product_type,
        selling_points,
        appearance_for_plan,
        proof_anchor,
        scenario,
        risk_level,
    )
    first_point = proof_plan.get("primary_value") or first_point
    second_point = proof_plan.get("result_value") or second_point
    feature_action = dict(feature_action)
    feature_action.update(
        {
            "visual": (
                f"同一件{product_type}保持完整外观，围绕「{first_point}」形成可见证据。"
                f"{proof_plan.get('source_action', '')}"
            ),
            "action": proof_plan.get("source_action") or feature_action.get("action", ""),
            "review_focus": _unique_list(_string_list(feature_action.get("review_focus")) + ["卖点是否由画面证明"]),
        }
    )
    scene_action = dict(scene_action)
    scene_action.update(
        {
            "visual": proof_plan.get("result_state") or scene_action.get("visual", ""),
            "action": proof_plan.get("result_action") or scene_action.get("action", ""),
            "review_focus": _unique_list(_string_list(scene_action.get("review_focus")) + ["结果状态是否证明卖点"]),
        }
    )

    if has_real_extension_anchor:
        opening_template = _v3_opening_source_scene_action_template(product_action, identity_card)
        storyboard = [
            _v3_product_shot(
                shot_index=0,
                role="product_reveal",
                duration=5,
                subtitle=first_point,
                scene_goal=f"从上传素材首帧建立真实商品和素材场景，并展示卖点：{first_point}",
                action_template=opening_template,
                product_type=product_type,
                identity_card=identity_card,
                asset_id=reveal_asset_id,
            ),
            _v3_product_shot(
                shot_index=1,
                role="feature_demo",
                duration=5,
                subtitle=second_point,
                scene_goal=f"用真实素材参考证明卖点：{second_point}",
                action_template=feature_action,
                product_type=product_type,
                identity_card=identity_card,
                asset_id=feature_asset_id,
            ),
            _v3_product_result_shot(
                shot_index=2,
                role="commerce_result",
                duration=5,
                subtitle=proof_plan.get("result_caption") or second_point,
                scene_goal=f"在商品仍然清楚可见的使用结果镜头中证明卖点：{second_point}",
                action_template=scene_action,
                product_type=product_type,
                identity_card=identity_card,
                asset_id=reveal_asset_id,
            ),
        ]
    else:
        storyboard = [
            _v3_context_shot(
                shot_index=0,
                role="hook",
                duration=5,
                subtitle=_context_problem_subtitle(product_type, first_point),
                scene_goal=_context_problem_goal(product_type, first_point),
                action_template=scene_action,
                product_type=product_type,
            ),
            _v3_product_shot(
                shot_index=1,
                role="product_reveal",
                duration=5,
                subtitle=first_point,
                scene_goal=f"从上传素材首帧建立真实商品身份，并展示卖点：{first_point}",
                action_template=product_action,
                product_type=product_type,
                identity_card=identity_card,
                asset_id=reveal_asset_id,
            ),
            _v3_product_shot(
                shot_index=2,
                role="feature_demo",
                duration=5,
                subtitle=second_point,
                scene_goal=f"用安全动作模板证明卖点：{second_point}",
                action_template=feature_action,
                product_type=product_type,
                identity_card=identity_card,
                asset_id=feature_asset_id,
            ),
        ]
    _attach_strategy_contract_to_storyboard(
        storyboard,
        strategy_id=_strategy_id_from_source(source),
        strategy_family=_strategy_family_from_source(source),
    )
    for shot in storyboard:
        shot["planner_source"] = source
        role = str(shot.get("narrative_role", "")).strip().lower()
        if role == "product_reveal" and has_real_extension_anchor:
            shot["material_strategy"] = "source_scene_extension"
            shot["selected_prompt_skill"] = "commerce_scene.source_confirm"
            shot["planner_source"] = f"{source}:material_first_source_scene_extension"
            shot["asset_usage"] = _v3_asset_usage_metadata(
                reveal_asset,
                profiles_by_id.get(reveal_asset_id),
                role="product_reveal",
                material_strategy="source_scene_extension",
                prompt_skill="commerce_scene.source_confirm",
            )
            shot["asset_usage_reason"] = shot["asset_usage"].get("asset_usage_reason", "")
        elif role == "feature_demo" and feature_asset and feature_asset_id:
            visual_role = str(feature_asset.get("visual_role", "")).strip()
            prompt_skill = (
                "detail_reference.static_feature_showcase"
                if visual_role in {"detail_reference", "logo_detail", "brand_detail"}
                else "source_scene_extension.static_feature_showcase"
            )
            shot["material_strategy"] = "detail_reference" if prompt_skill.startswith("detail_reference") else "source_scene_extension"
            shot["selected_prompt_skill"] = prompt_skill
            shot["asset_usage"] = _v3_asset_usage_metadata(
                feature_asset,
                profiles_by_id.get(feature_asset_id),
                role="feature_demo",
                material_strategy=shot["material_strategy"],
                prompt_skill=prompt_skill,
            )
            shot["asset_usage_reason"] = shot["asset_usage"].get("asset_usage_reason", "")
            if visual_role in {"detail_reference", "logo_detail", "brand_detail"}:
                shot["shot_type"] = "特写"
                shot["visual_description"] = (
                    f"{feature_action['visual']}。本镜使用细节参考素材展示卖点：{second_point}。"
                    "只展示素材中已经清楚可见的结构、标识或材质细节，不凭空补全整机新角度。"
                )
                shot["asset_requirement"] = "优先使用 detail_reference 细节素材；只承担卖点/细节证明，不作为完整外观重建依据。"
        elif role == "commerce_result" and reveal_asset and reveal_asset_id:
            shot["material_strategy"] = "source_scene_extension"
            shot["selected_prompt_skill"] = "source_scene_extension.product_result_scene"
            shot["asset_usage"] = _v3_asset_usage_metadata(
                reveal_asset,
                profiles_by_id.get(reveal_asset_id),
                role="commerce_result",
                material_strategy="source_scene_extension",
                prompt_skill="source_scene_extension.product_result_scene",
            )
            shot["asset_usage_reason"] = shot["asset_usage"].get("asset_usage_reason", "")
    for shot in storyboard:
        if str(shot.get("product_presence", "")).strip().lower() == "forbidden":
            shot["asset_requirement"] = f"无商品剧情镜头：走文生视频，禁止出现{product_type}、同类商品、logo 或品牌文字。"
            shot["completion_criteria"] = [
                "剧情场景清楚表达使用痛点或生活情境",
                "画面中不出现待售商品或同类商品",
                "不凭空生成可识别品牌、logo 或商品文字",
            ]
        else:
            shot["asset_requirement"] = shot.get("asset_requirement") or "必须使用上传商品图作为首帧锚点；禁止文生视频生成可识别商品。"
            shot["completion_criteria"] = [
                "商品主体来自真实素材首帧",
                "商品结构、颜色、标识和主体数量保持稳定",
                "动作只执行本镜 action_template 中的单一动作",
            ]
            shot["continuity_group"] = ""
            shot["transition_type"] = "hard_cut"
            shot["anchor_last_frame"] = False
            appearance = str(identity_card.get("appearance_summary", "")).strip() or f"上传素材中的同一件{product_type}"
            shot["video_prompt"] = _compose_final_video_prompt(shot, product_type=product_type, appearance=appearance)
            shot["force_video_prompt"] = True
            shot["final_prompt_source"] = "strategy_contract"
    _attach_strategy_contract_to_storyboard(
        storyboard,
        strategy_id=_strategy_id_from_source(source),
        strategy_family=_strategy_family_from_source(source),
    )
    return storyboard


def _render_ab_variants(
    *,
    task_id: str,
    task_data: dict[str, Any],
    product_context: dict[str, Any],
    asset_analysis: dict[str, Any],
    script_review_variants: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """生成不覆盖默认结果的候选策略视频。"""

    if os.getenv("AIGC_DISABLE_AB_VARIANTS") == "1":
        return {}

    variant_id = "B_ideal_commerce_scene"
    output_dir = Path(_task_output_dir(task_data, task_id)) / "variants" / variant_id
    started_at = time.perf_counter()
    try:
        review_variant = _safe_dict((script_review_variants or {}).get(variant_id, {}))
        storyboard = list(review_variant.get("storyboard") or [])
        if not storyboard:
            storyboard = _plan_ideal_commerce_scene_storyboard(product_context, asset_analysis)
        if not storyboard:
            return {
                variant_id: {
                    "strategy": variant_id,
                    "success": False,
                    "error": "no_ideal_commerce_storyboard",
                    "review_notes": ["没有足够商品信息生成 B 候选。"],
                }
            }
        asset_matching = match_assets_to_storyboard(storyboard, asset_analysis)
        asset_gap_completion = complete_asset_gaps(
            storyboard=storyboard,
            asset_matching=asset_matching,
            asset_analysis=asset_analysis,
            product_identity_card=product_context.get("product_identity_card", {}),
        )
        creation_plan = build_creation_plan(
            product_context,
            storyboard,
            asset_gap_completion.get("asset_matching", asset_matching),
        )
        creation_plan["variant_strategy"] = variant_id
        creation_plan["diagnostic_mode"] = "ideal_commerce_scene_no_auto_repair"
        creation_plan["review_policy"] = "B 变体只生成和记录，不进入自动内容修复，避免诊断样本被二次改写。"
        render_result = render_seedance_video(
            task_id=task_id,
            creation_plan=creation_plan,
            output_dir=str(output_dir),
        )
        return {
            variant_id: {
                "strategy": variant_id,
                "success": bool(render_result.get("success")),
                "storyboard": storyboard,
                "script_plan": review_variant.get("script_plan", {}),
                "asset_matching": asset_gap_completion.get("asset_matching", asset_matching),
                "asset_gap_completion": asset_gap_completion,
                "creation_plan": creation_plan,
                "render_result": render_result,
                "video_path": str(render_result.get("video_path", "")),
                "review_notes": [
                    "B 是理想带货场景候选，不覆盖默认 A。",
                    "B 优先把卖点转成可见动作或结果状态，允许更高风险，用来定位模型能力边界。",
                    "B 暂不做自动修复；如果失败，应查看每镜 prompt、素材绑定和视频结果之间的差距。",
                ],
                "elapsed_seconds": round(time.perf_counter() - started_at, 2),
            }
        }
    except Exception as exc:
        return {
            variant_id: {
                "strategy": variant_id,
                "success": False,
                "error": str(exc),
                "video_path": "",
                "review_notes": ["B 候选生成失败，但默认 A 已保留。"],
                "elapsed_seconds": round(time.perf_counter() - started_at, 2),
            }
        }


def _prompt_skill_template_path(skill_id: str) -> Path:
    """Map a stable prompt skill id to its markdown template file."""

    safe_parts = [
        re.sub(r"[^a-zA-Z0-9_-]", "", part)
        for part in str(skill_id).strip().split(".")
        if part.strip()
    ]
    if len(safe_parts) < 2:
        return PROMPT_SKILL_LIBRARY_DIR / "_invalid_skill_id_.md"
    return PROMPT_SKILL_LIBRARY_DIR.joinpath(*safe_parts[:-1], f"{safe_parts[-1]}.md")


def _extract_prompt_skill_template(markdown_text: str) -> str:
    """Extract the final prompt template body from a skill markdown file."""

    text = str(markdown_text or "")
    match = re.search(r"(?ms)^##\s*Prompt\s*模板\s*\n(?P<body>.*?)(?=^##\s+|\Z)", text)
    body = match.group("body") if match else text
    body = body.strip()
    if body.startswith("```"):
        body = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", body)
        body = re.sub(r"\s*```$", "", body)
    return body.strip()


def _load_prompt_skill_template(skill_id: str) -> str:
    """Load a markdown prompt skill template, cached by skill id."""

    normalized_id = str(skill_id or "").strip()
    if not normalized_id:
        return ""
    cached = _PROMPT_SKILL_TEMPLATE_CACHE.get(normalized_id)
    if cached is not None:
        return cached
    path = _prompt_skill_template_path(normalized_id)
    try:
        template = _extract_prompt_skill_template(path.read_text(encoding="utf-8"))
    except OSError:
        template = ""
    _PROMPT_SKILL_TEMPLATE_CACHE[normalized_id] = template
    return template


def _load_prompt_skill_reference(relative_path: str, *, max_chars: int = 18000) -> str:
    """Load a prompt-skill reference document for upstream LLM planning."""

    safe_parts = [
        part
        for part in Path(str(relative_path or "")).parts
        if part not in {"", ".", ".."} and not part.startswith("/")
    ]
    if not safe_parts:
        return ""
    normalized_path = "/".join(safe_parts)
    cached = _PROMPT_SKILL_REFERENCE_CACHE.get(normalized_path)
    if cached is not None:
        return cached
    path = PROMPT_SKILL_LIBRARY_DIR.joinpath(*safe_parts)
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip()
    _PROMPT_SKILL_REFERENCE_CACHE[normalized_path] = text
    return text


def _render_prompt_skill_template(
    skill_id: str,
    variables: dict[str, Any],
    fallback: str,
) -> str:
    """Render a prompt skill template with a safe code fallback."""

    template = _load_prompt_skill_template(skill_id) or str(fallback or "")
    cleaned_variables = {
        key: _stringify_prompt_variable(value)
        for key, value in _safe_dict(variables).items()
    }

    def replace(match: re.Match[str]) -> str:
        key = match.group(1) or match.group(2) or ""
        return cleaned_variables.get(key, "")

    rendered = _PROMPT_SKILL_PLACEHOLDER_RE.sub(replace, template)
    rendered = re.sub(r"\n{3,}", "\n\n", rendered)
    rendered = re.sub(r"[ \t]{2,}", " ", rendered)
    return rendered.strip()


def _stringify_prompt_variable(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            item_text = _stringify_prompt_variable(item)
            if item_text:
                parts.append(f"{key}：{item_text}")
        return "；".join(parts)
    if isinstance(value, (list, tuple, set)):
        return "、".join(str(item).strip() for item in value if str(item).strip())
    return str(value).strip()


_COMMERCE_EXPRESSION_STRATEGIES = {
    "direct_benefit_proof",
    "usage_result_demo",
    "premium_texture_reveal",
    "scene_fit_showcase",
    "feature_operation_demo",
    "problem_solution_pair",
    "aspirational_lifestyle_result",
    "identity_material_confirm",
    "source_context_extension",
    "comparison_contrast",
    "trust_source_evidence",
    "sensory_use_cue",
    "gift_unboxing_reveal",
    "routine_integration",
}


def _contains_commerce_strategy_label(text: str) -> bool:
    normalized = str(text or "")
    return any(strategy in normalized for strategy in _COMMERCE_EXPRESSION_STRATEGIES)


def _clean_commerce_plan_text(
    value: Any,
    *,
    fallback: str,
    max_chars: int = 180,
    reject_strategy_labels: bool = True,
) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.strip("`#*- \t\r\n")
    if not text:
        text = fallback
    if reject_strategy_labels and _contains_commerce_strategy_label(text):
        text = fallback
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip("，,、；;。.!！?？ ")
    return text or fallback


def _commerce_asset_summary(asset_analysis: dict[str, Any], anchor: dict[str, Any]) -> dict[str, Any]:
    asset_profiles = _safe_dict(asset_analysis).get("asset_profiles", [])
    if not isinstance(asset_profiles, list):
        asset_profiles = []
    anchor_id = str(anchor.get("asset_id", "")).strip()
    anchor_profile = {}
    for profile in asset_profiles:
        if isinstance(profile, dict) and str(profile.get("asset_id", "")).strip() == anchor_id:
            anchor_profile = profile
            break
    return {
        "selected_anchor": {
            "asset_id": anchor_id,
            "visual_role": anchor.get("visual_role", ""),
            "quality_score": anchor.get("quality_score", ""),
            "product_visibility": anchor.get("product_visibility", ""),
            "role_source": anchor.get("role_source", ""),
        },
        "anchor_profile": {
            "visual_role": anchor_profile.get("visual_role", ""),
            "suitable_for": _string_list(anchor_profile.get("suitable_for")),
            "material_capabilities": _safe_dict(anchor_profile.get("material_capabilities")),
            "usage_risks": _string_list(anchor_profile.get("usage_risks")),
            "reasoning": anchor_profile.get("reasoning", ""),
        },
        "asset_capability_plan": _safe_dict(asset_analysis.get("asset_capability_plan")),
        "asset_selection_diagnostics": [
            {
                "asset_id": item.get("asset_id", ""),
                "visual_role": item.get("visual_role", ""),
                "role_source": item.get("role_source", ""),
                "reason": item.get("reason", ""),
            }
            for item in asset_analysis.get("asset_selection_diagnostics", [])
            if isinstance(item, dict)
        ][:4],
    }


def _commerce_expression_plan_prompt(
    *,
    product_context: dict[str, Any],
    identity_card: dict[str, Any],
    product_type: str,
    selling_points: list[str],
    appearance: str,
    scenario: dict[str, str],
    asset_summary: dict[str, Any],
    risk_level: str,
) -> dict[str, Any]:
    strategy_reference = _load_prompt_skill_reference("shared/commerce_expression_strategies.md", max_chars=22000)
    anti_patterns = _load_prompt_skill_reference("shared/anti_patterns.md", max_chars=9000)
    return {
        "task": "为商品带货视频选择通用商业表达结构，并生成三镜的价值证明计划。",
        "hard_rules": [
            "不要把商品类型写成固定 if/else；同一商品也可能因为用户目标不同选择不同表达结构。",
            "不要默认使用痛点-解决。只有痛点和商品解决动作都能被画面证明时才选择 problem_solution_pair。",
            "可以选择优势展示、质感展示、操作证明、结果展示、场景适配、生活方式结果、信任证据等任意合适结构。",
            "素材理解先于剧本。剧本和动作必须顺着素材场景、商品结构、标识风险和可拍能力展开，不能和素材场景冲突。",
            "每个 5 秒镜头只承担一个清楚表达目标；不要把跨地点、拿起、放入、走路、展示细节等多个动作塞进一个镜头。",
            "source_action 必须服务于具体卖点表达，动作是否成立由素材、商品结构和 skill 样例共同决定。",
            "最终给视频模型的 prompt 不能包含内部策略 id、评分、JSON 或决策推理。这里输出 JSON 只给系统内部使用。",
            "字幕必须是用户可见的短句，不能输出“看看这个好物”“点击了解更多”“真实外观确认”“真实体验”。",
        ],
        "expected_json_shape": {
            "expression_strategy": "从 strategy_reference 中选择一个最合适的策略 id",
            "primary_value": "第一/第二镜要证明的具体商品优势，短句",
            "result_value": "第三镜要证明的使用结果或场景价值，短句",
            "confirm_caption": "第一镜字幕，短句，不是内部标签",
            "action_caption": "第二镜字幕，短句，不是内部标签",
            "result_caption": "第三镜字幕，短句，不是内部标签",
            "source_place": "素材场景中合理的地点描述，不和素材冲突",
            "result_place": "新场景结果镜的具体地点描述",
            "human": "人物参与方式，尽量不抢商品主体",
            "source_action": "第二镜在素材首帧场景里可完成的一个动作或结果状态，必须能从画面证明 primary_value",
            "result_action": "第三镜新场景中的轻微辅助动作，不跨地点",
            "result_state": "第三镜开头已经成立的商品+场景+结果关系，能证明 result_value",
            "notes_for_review": "内部简短理由，不能进入视频 prompt",
        },
        "product_context": {
            "title": product_context.get("product_title") or product_context.get("title"),
            "usage_scene": product_context.get("usage_scene"),
            "target_audience": product_context.get("target_audience") or product_context.get("audience"),
            "structured_requirements": _safe_dict(product_context.get("structured_requirements")),
            "selling_points": selling_points,
        },
        "material_dossier": {
            "product_type": product_type,
            "appearance": appearance,
            "identity_card": identity_card,
            "asset_summary": asset_summary,
            "default_scenario_if_needed": scenario,
            "reconstruction_risk": risk_level,
        },
        "strategy_reference": strategy_reference,
        "anti_patterns_reference": anti_patterns,
    }


def _parse_commerce_expression_plan(raw_content: str) -> dict[str, Any] | None:
    parsed = _extract_json_from_text(str(raw_content or ""))
    if parsed is None:
        repaired = _repair_json_response(
            str(raw_content or ""),
            "commerce_expression_plan",
            expected_shape={
                "expression_strategy": "string",
                "primary_value": "string",
                "result_value": "string",
                "confirm_caption": "string",
                "action_caption": "string",
                "result_caption": "string",
                "source_action": "string",
                "result_action": "string",
                "result_state": "string",
            },
        )
        parsed = repaired.get("parsed") if repaired.get("ok") else None
    if isinstance(parsed, dict):
        return parsed
    return None


def _normalize_commerce_expression_plan(
    raw_plan: dict[str, Any],
    *,
    fallback_plan: dict[str, str],
    scenario: dict[str, str],
) -> dict[str, str]:
    strategy = str(raw_plan.get("expression_strategy") or "").strip()
    if strategy not in _COMMERCE_EXPRESSION_STRATEGIES:
        strategy = fallback_plan.get("expression_strategy", "direct_benefit_proof")

    primary_value = _clean_short_sentence(
        raw_plan.get("primary_value") or fallback_plan.get("primary_value", ""),
        max_chars=18,
    ) or fallback_plan.get("primary_value", "卖点看得见")
    result_value = _clean_short_sentence(
        raw_plan.get("result_value") or fallback_plan.get("result_value", ""),
        max_chars=18,
    ) or primary_value
    confirm_caption = _safe_user_caption(
        str(raw_plan.get("confirm_caption") or fallback_plan.get("confirm_caption", "")),
        fallback=primary_value,
        max_chars=DEFAULT_SUBTITLE_MAX_CHARS,
    )
    action_caption = _safe_user_caption(
        str(raw_plan.get("action_caption") or fallback_plan.get("action_caption", "")),
        fallback=primary_value,
        max_chars=DEFAULT_SUBTITLE_MAX_CHARS,
    )
    result_caption = _safe_user_caption(
        str(raw_plan.get("result_caption") or fallback_plan.get("result_caption", "")),
        fallback=result_value,
        max_chars=DEFAULT_SUBTITLE_MAX_CHARS,
    )

    source_action = _clean_commerce_plan_text(
        raw_plan.get("source_action"),
        fallback=fallback_plan.get("source_action", "商品在素材场景中保持清楚可见，画面证明一个具体卖点。"),
        max_chars=220,
    )
    result_action = _clean_commerce_plan_text(
        raw_plan.get("result_action"),
        fallback=fallback_plan.get("result_action", "人物只做轻微辅助动作，让画面结果证明卖点。"),
        max_chars=220,
    )
    result_state = _clean_commerce_plan_text(
        raw_plan.get("result_state"),
        fallback=fallback_plan.get("result_state", "商品处在真实使用位置，和周边道具形成清楚关系。"),
        max_chars=240,
    )
    source_place = _clean_commerce_plan_text(
        raw_plan.get("source_place"),
        fallback=scenario.get("source_place", "素材原始场景"),
        max_chars=40,
        reject_strategy_labels=False,
    )
    result_place = _clean_commerce_plan_text(
        raw_plan.get("result_place"),
        fallback=scenario.get("result_place", "真实使用场景"),
        max_chars=50,
        reject_strategy_labels=False,
    )
    human = _clean_commerce_plan_text(
        raw_plan.get("human"),
        fallback=scenario.get("human", "普通用户的手部或半身动作"),
        max_chars=70,
        reject_strategy_labels=False,
    )

    return {
        "expression_strategy": strategy,
        "primary_value": primary_value,
        "result_value": result_value,
        "confirm_caption": confirm_caption,
        "action_caption": action_caption,
        "result_caption": result_caption,
        "source_action": source_action,
        "result_action": result_action,
        "result_state": result_state,
        "source_place": source_place,
        "result_place": result_place,
        "human": human,
    }


def _commerce_expression_plan(
    product_context: dict[str, Any],
    asset_analysis: dict[str, Any],
    identity_card: dict[str, Any],
    product_type: str,
    selling_points: list[str],
    appearance: str,
    anchor: dict[str, Any],
    scenario: dict[str, str],
    risk_level: str,
) -> dict[str, str]:
    """Ask the LLM to choose a general commerce expression plan; rules are fallback only."""

    usage_scene = str(
        product_context.get("usage_scene")
        or _safe_dict(product_context.get("structured_requirements")).get("usage_scene")
        or ""
    ).strip()
    if "杯" in product_type:
        fallback_plan = build_value_proof_plan(
            product_type=product_type,
            selling_points=selling_points,
            usage_scene=usage_scene,
            material_risk=risk_level,
        )
    else:
        fallback_plan = _commerce_visual_proof_plan(product_context, identity_card, product_type, selling_points)
    fallback_plan = ensure_value_proof_plan(
        fallback_plan,
        product_type=product_type,
        selling_points=selling_points,
        usage_scene=usage_scene,
        material_risk=risk_level,
    )
    fallback_plan["expression_plan_source"] = "fallback_rule_plan"

    if os.getenv("PYTEST_CURRENT_TEST") and os.getenv("AIGC_TEST_ALLOW_LLM") != "1":
        return fallback_plan

    prompt_data = _commerce_expression_plan_prompt(
        product_context=product_context,
        identity_card=identity_card,
        product_type=product_type,
        selling_points=selling_points,
        appearance=appearance,
        scenario=scenario,
        asset_summary=_commerce_asset_summary(asset_analysis, anchor),
        risk_level=risk_level,
    )
    llm_result = _call_text_llm(prompt_data, purpose="commerce_expression_plan", temperature=0.35)
    if not llm_result.get("ok"):
        fallback_plan["expression_plan_error"] = str(llm_result.get("error", "llm_failed"))
        return fallback_plan

    raw_plan = _parse_commerce_expression_plan(str(llm_result.get("content", "")))
    if not raw_plan:
        fallback_plan["expression_plan_error"] = "llm_plan_parse_failed"
        return fallback_plan

    normalized = _normalize_commerce_expression_plan(raw_plan, fallback_plan=fallback_plan, scenario=scenario)
    normalized = ensure_value_proof_plan(
        normalized,
        product_type=product_type,
        selling_points=selling_points,
        usage_scene=usage_scene,
        material_risk=risk_level,
    )
    normalized["expression_plan_source"] = "llm_skill_plan"
    normalized["expression_plan_review_notes"] = _clean_commerce_plan_text(
        raw_plan.get("notes_for_review"),
        fallback="LLM 根据素材理解、用户目标和策略库选择表达结构。",
        max_chars=160,
        reject_strategy_labels=False,
    )
    return normalized


def _plan_ideal_commerce_scene_storyboard(
    product_context: dict[str, Any],
    asset_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """B 候选：用少量高质量单场景 prompt 测试更理想的带货剧情。"""

    identity_card = _safe_dict(product_context.get("product_identity_card", {}))
    product_type = str(identity_card.get("product_type") or product_context.get("product_title") or "商品").strip()
    selling_points = _selling_point_phrases(product_context.get("selling_points", []))
    appearance = _commerce_appearance_text(identity_card, product_type)
    anchor = _find_best_appearance_anchor(asset_analysis) or {}
    asset_id = str(anchor.get("asset_id", "")).strip()
    style = _commerce_style_text(product_context.get("visual_style_bible"))
    scenario = _commerce_scenario_from_context(product_context, identity_card, product_type)
    risk_level = _commerce_reconstruction_risk(identity_card, anchor)
    proof_plan = _commerce_expression_plan(
        product_context,
        asset_analysis,
        identity_card,
        product_type,
        selling_points,
        appearance,
        anchor,
        scenario,
        risk_level,
    )
    scenario = dict(scenario)
    scenario["source_place"] = proof_plan.get("source_place", scenario.get("source_place", "素材原始场景"))
    scenario["result_place"] = proof_plan.get("result_place", scenario.get("result_place", "真实使用场景"))
    scenario["human"] = proof_plan.get("human", scenario.get("human", "普通用户的手部或半身动作"))

    if not asset_id:
        return []

    first_point = proof_plan["primary_value"]
    second_point = proof_plan["result_value"]
    prompt_skill_ids = [
        "commerce_scene.source_confirm",
        "commerce_scene.material_action_proof",
        "commerce_scene.new_scene_result",
    ]
    confirm_prompt = _commerce_prompt_source_confirm(
        product_type=product_type,
        appearance=appearance,
        style=style,
        scenario=scenario,
    )
    action_prompt = _commerce_prompt_material_action(
        product_type=product_type,
        appearance=appearance,
        style=style,
        proof_plan=proof_plan,
        first_point=first_point,
    )
    result_prompt = _commerce_prompt_new_scene_result(
        product_type=product_type,
        appearance=appearance,
        style=style,
        scenario=scenario,
        proof_plan=proof_plan,
        second_point=second_point,
        risk_level=risk_level,
    )
    prompts = [confirm_prompt, action_prompt, result_prompt]
    roles = ["product_confirm", "commerce_action_proof", "commerce_result_scene"]
    subtitles = _commerce_subtitles(proof_plan)
    storyboard: list[dict[str, Any]] = []
    for index, prompt_text in enumerate(prompts):
        is_text_result = index == 2
        shot = {
            "shot_index": index,
            "duration_seconds": 5,
            "narrative_role": roles[index],
            "purpose": [
                "先从素材确认商品身份，避免后续剧情变成其他商品。",
                f"用一个清晰动作证明卖点：{first_point}。",
                f"硬切到新场景，用结果状态证明卖点：{second_point}。",
            ][index],
            "scene_goal": [
                "建立真实商品和素材外观。",
                "在素材基础上做一次更大胆但边界清楚的商品动作扩展。",
                "用独立生活场景表达带货结果，不承接上一镜时间空间。",
            ][index],
            "initial_state": "本镜头第一帧是上传素材中的同一件商品。" if not is_text_result else "这是硬切后的新镜头，时间和空间都从新场景开始。",
            "action": [
                "0-1 秒保持首帧商品稳定，之后只做轻微光线和镜头呼吸。",
                proof_plan["source_action"],
                proof_plan["result_action"],
            ][index],
            "final_state": [
                "观众能确认商品外观、颜色、结构和标识区域。",
                "动作结束后商品仍保持同一件商品身份。",
                "观众能从画面结果理解商品的带货卖点。",
            ][index],
            "camera_motion": "定镜或非常轻微的生活感手持运动",
            "visual_description": prompt_text,
            "subtitle": subtitles[index],
            "voiceover": subtitles[index],
            "asset_id": "" if is_text_result else asset_id,
            "asset_requirement": "使用上传素材作为首帧锚点。" if not is_text_result else "不使用首帧素材，靠详细外观描述生成新场景结果镜，用于诊断理想剧情上限。",
            "render_strategy": "text_to_video" if is_text_result else "image_to_video",
            "product_presence": "optional" if is_text_result else "required",
            "identity_strictness": "medium" if is_text_result else "high",
            "transition_type": "hard_cut",
            "continuity_group": "",
            "anchor_last_frame": False,
            "allowed_variation": ["自然光线变化", "人物手部自然动作", "真实生活场景变化"],
            "forbidden_variation": _unique_list(
                _string_list(identity_card.get("forbidden_changes"))
                + ["错误商品类型", "多个同类主商品", "新增非商品自带文字或 UI", "无物理接触的自动移动"]
            ),
            "review_focus": ["商品身份", "动作是否完成", "卖点是否由画面证明", "物理关系是否合理"],
            "completion_criteria": [
                "单镜头只有一个主要动作或一个结果状态",
                "商品外观与素材理解一致",
                "场景边界清楚，不把多个时间地点混进同一镜头",
            ],
            "product_identity_card": identity_card,
            "planner_source": "B_ideal_commerce_scene:single_scene_prompt_montage",
            "material_strategy": "ideal_commerce_scene",
            "expression_strategy": proof_plan.get("expression_strategy", "direct_benefit_proof"),
            "expression_plan_source": proof_plan.get("expression_plan_source", ""),
            "selected_prompt_skill": prompt_skill_ids[index],
            "required_for_variant": True,
            "force_video_prompt": True,
            "video_prompt": prompt_text,
            "seedance_prompt": prompt_text,
            "risk_notes": [
                "B 候选允许测试更大胆场景，可能出现商品身份漂移；用于和 A 保守版对比。",
                f"reconstruction_risk={risk_level}",
            ],
        }
        storyboard.append(shot)
    return storyboard


def _commerce_appearance_text(identity_card: dict[str, Any], product_type: str) -> str:
    parts = [
        str(identity_card.get("appearance_summary", "")).strip(),
        "主色：" + str(identity_card.get("primary_color", "")).strip() if identity_card.get("primary_color") else "",
        "可见标识：" + "、".join(_string_list(identity_card.get("visible_marks"))) if identity_card.get("visible_marks") else "",
        "关键结构：" + "、".join(_string_list(identity_card.get("key_components"))) if identity_card.get("key_components") else "",
        "材质：" + "、".join(_string_list(identity_card.get("material_features"))) if identity_card.get("material_features") else "",
    ]
    text = "；".join(part for part in parts if part)
    return text or f"上传素材中清楚可见的同一件{product_type}"


def _commerce_style_text(raw_style: Any) -> str:
    if isinstance(raw_style, dict):
        summary = str(raw_style.get("style_summary") or raw_style.get("user_style") or "").strip()
        if summary:
            return f"真实写实的商业短视频；{summary}；风格统一，背景不抢商品主体"
        values = [
            str(raw_style.get(key, "")).strip()
            for key in ("realism", "lighting", "color_temperature", "background_complexity", "camera_language", "user_style")
        ]
        text = "；".join(value for value in values if value)
        if text:
            return text
    return "真实写实的商业短视频；柔和自然光；稳定镜头；背景真实但不抢商品主体"


def _commerce_scenario_from_context(
    product_context: dict[str, Any],
    identity_card: dict[str, Any],
    product_type: str,
) -> dict[str, str]:
    usage_scene = str(
        product_context.get("usage_scene")
        or _safe_dict(product_context.get("structured_requirements")).get("usage_scene")
        or ""
    )
    audience = str(
        product_context.get("target_audience")
        or product_context.get("audience")
        or _safe_dict(product_context.get("structured_requirements")).get("target_audience")
        or "日常用户"
    )
    combined = usage_scene + " " + audience + " " + " ".join(_string_list(identity_card.get("functional_features")))
    if any(word in combined for word in ("护肤", "美妆", "梳妆", "浴室", "洗漱", "面霜", "精华")):
        return {
            "name": "护理梳妆",
            "source_place": "自然光梳妆台或浴室台面",
            "result_place": "整洁梳妆台、浴室洗漱台或晚间护理桌面",
            "human": "普通用户的手部和镜前护理动作，不抢商品主体",
        }
    if any(word in combined for word in ("厨房", "料理", "烹饪", "餐桌", "清洁", "家务", "小家电", "居家")):
        return {
            "name": "居家使用",
            "source_place": "自然光厨房台面或居家桌面",
            "result_place": "整洁厨房台面、餐桌边或居家使用位置",
            "human": "普通用户的手部和轻微家务动作，不抢商品主体",
        }
    if any(word in combined for word in ("穿搭", "服饰", "上身", "衣柜", "通勤装", "搭配")):
        return {
            "name": "穿搭展示",
            "source_place": "自然光衣柜旁、床边或整理台面",
            "result_place": "玄关镜前、衣柜旁或通勤出门前的穿搭场景",
            "human": "普通用户的手部、半身或衣物整理动作，不抢商品主体",
        }
    if any(word in combined for word in ("通勤", "出门", "办公室", "上班", "背包", "便携")):
        return {
            "name": "通勤携带",
            "source_place": "清晨室内桌面或玄关台面",
            "result_place": "办公楼入口、通勤背包旁或办公室工位",
            "human": "穿浅色外套的通勤者，只露出手部、手臂或半身，不抢商品主体",
        }
    if any(word in combined for word in ("学习", "宿舍", "图书馆", "学生")):
        return {
            "name": "学习桌面",
            "source_place": "明亮书桌",
            "result_place": "图书馆自习桌或宿舍书桌",
            "human": "学生的手部和书本电脑等普通学习道具",
        }
    return {
        "name": "日常使用",
        "source_place": "自然光桌面",
        "result_place": "整洁生活空间或办公桌",
        "human": "普通用户的手部或半身动作",
    }


def _commerce_strategy_keywords(strategy: str) -> tuple[str, ...]:
    keywords_by_strategy = {
        "direct_benefit_proof": ("省心", "方便", "高效", "舒适", "耐用", "稳定", "核心", "卖点"),
        "usage_result_demo": ("结果", "省心", "方便", "舒适", "清爽", "保湿", "滋润", "整洁", "持久", "效果"),
        "premium_texture_reveal": ("材质", "质感", "纹理", "肌理", "磨砂", "金属", "皮革", "玻璃", "面料", "亲肤", "柔软", "细腻", "丝滑", "光泽", "高级", "礼盒"),
        "scene_fit_showcase": ("便携", "收纳", "通勤", "出门", "旅行", "办公", "居家", "厨房", "浴室", "梳妆", "穿搭", "搭配", "包内", "桌面"),
        "feature_operation_demo": ("接口", "按键", "按钮", "开关", "旋钮", "泵头", "喷头", "拉链", "卡扣", "磁吸", "开合", "打开", "折叠", "展开", "调节", "一键", "档位"),
        "problem_solution_pair": ("解决", "告别", "不再", "避免", "改善", "减少", "缓解", "凌乱", "干燥", "卡顿", "费力", "难"),
        "aspirational_lifestyle_result": ("仪式感", "精致", "颜值", "送礼", "礼盒", "高级", "体面", "氛围", "悦己"),
        "identity_material_confirm": ("标识", "logo", "结构", "材质", "颜色", "细节", "纹理", "包装"),
    }
    return keywords_by_strategy.get(strategy, ())


def _commerce_pick_specific_value(
    terms: list[str],
    keywords: tuple[str, ...],
    *,
    fallback_terms: list[str],
    fallback: str,
) -> str:
    cleaned_terms = [
        _clean_short_sentence(term, max_chars=18)
        for term in terms + fallback_terms
        if str(term).strip()
    ]
    deduped_terms = []
    for term in cleaned_terms:
        if term and term not in deduped_terms and not _is_internal_or_generic_caption(term):
            deduped_terms.append(term)
    for term in deduped_terms:
        if any(keyword in term for keyword in keywords):
            return term
    return deduped_terms[0] if deduped_terms else fallback


def _commerce_scene_phrase(usage_scene: str, target_audience: str) -> str:
    usage = _clean_short_sentence(usage_scene, max_chars=10)
    audience = _clean_short_sentence(target_audience, max_chars=8)
    if usage and audience and audience not in usage:
        return f"{audience}{usage}"
    return usage or audience or "日常使用场景"


def _commerce_pick_component(key_components: list[str], operation_words: tuple[str, ...]) -> str:
    for component in key_components:
        if any(word in component for word in operation_words):
            return _clean_short_sentence(component, max_chars=10) or "功能部件"
    if key_components:
        return _clean_short_sentence(key_components[0], max_chars=10) or "功能部件"
    return "功能部件"


def _commerce_pick_operation_action(allowed_actions: list[str], component: str) -> str:
    for action in allowed_actions:
        cleaned = _clean_short_sentence(action, max_chars=18)
        if cleaned:
            return cleaned
    return f"根据{component}的真实结构选择一个可拍操作"


def _commerce_visual_proof_plan(
    product_context: dict[str, Any],
    identity_card: dict[str, Any],
    product_type: str,
    selling_points: list[str],
) -> dict[str, str]:
    motion_affordance = _safe_dict(identity_card.get("motion_affordance"))
    can_handheld = bool(motion_affordance.get("can_be_handheld", True))
    visible_marks = _string_list(identity_card.get("visible_marks"))
    key_components = _string_list(identity_card.get("key_components"))
    material_features = _string_list(identity_card.get("material_features"))
    functional_features = _unique_list(
        _string_list(identity_card.get("functional_features"))
        + _string_list(product_context.get("functional_features"))
    )
    allowed_actions = _string_list(motion_affordance.get("allowed_actions"))
    usage_scene = str(
        product_context.get("usage_scene")
        or _safe_dict(product_context.get("structured_requirements")).get("usage_scene")
        or ""
    ).strip()
    target_audience = str(
        product_context.get("target_audience")
        or product_context.get("audience")
        or _safe_dict(product_context.get("structured_requirements")).get("target_audience")
        or ""
    ).strip()
    proof_terms = selling_points + functional_features + material_features + key_components
    context_terms = proof_terms + visible_marks + allowed_actions + [usage_scene, target_audience]
    text = " ".join(context_terms)

    material_words = (
        "材质", "质感", "纹理", "肌理", "磨砂", "金属", "皮革", "玻璃", "陶瓷", "棉", "羊毛",
        "面料", "亲肤", "柔软", "细腻", "丝滑", "光泽", "透明", "哑光", "高级", "礼盒", "包装",
    )
    operation_words = (
        "接口", "按键", "按钮", "开关", "旋钮", "泵头", "喷头", "拉链", "卡扣", "磁吸", "开合",
        "打开", "闭合", "折叠", "展开", "抽拉", "调节", "一键", "充电", "指示灯", "出风", "加热",
        "安装", "操作", "切换", "档位",
    )
    scene_words = (
        "便携", "收纳", "通勤", "出门", "旅行", "户外", "露营", "办公室", "办公", "宿舍", "居家",
        "厨房", "浴室", "梳妆", "穿搭", "搭配", "上身", "车载", "桌面", "包内", "背包",
    )
    problem_words = ("解决", "告别", "不再", "避免", "防止", "改善", "减少", "缓解", "凌乱", "干燥", "卡顿", "费力", "尴尬", "难")
    result_words = (
        "省心", "高效", "效率", "舒适", "舒服", "清爽", "保湿", "滋润", "修护", "提亮", "稳定",
        "整洁", "顺滑", "持久", "方便", "轻松", "快速", "耐用", "容量", "大容量", "效果",
    )
    lifestyle_words = ("仪式感", "精致", "颜值", "送礼", "礼盒", "高级", "体面", "氛围", "悦己", "质感生活")

    def has_any(words: tuple[str, ...]) -> bool:
        return any(word in text for word in words)

    scores = {
        "direct_benefit_proof": 1,
        "usage_result_demo": 0,
        "premium_texture_reveal": 0,
        "scene_fit_showcase": 0,
        "feature_operation_demo": 0,
        "problem_solution_pair": 0,
        "aspirational_lifestyle_result": 0,
        "identity_material_confirm": 0,
    }
    if has_any(material_words):
        scores["premium_texture_reveal"] += 4
        scores["identity_material_confirm"] += 1
    if has_any(operation_words) or allowed_actions:
        scores["feature_operation_demo"] += 4
    if has_any(scene_words) or usage_scene or target_audience:
        scores["scene_fit_showcase"] += 3
        scores["usage_result_demo"] += 1
    if has_any(problem_words):
        scores["problem_solution_pair"] += 4
    if has_any(result_words):
        scores["usage_result_demo"] += 3
        scores["direct_benefit_proof"] += 1
    if has_any(lifestyle_words):
        scores["aspirational_lifestyle_result"] += 4
        scores["premium_texture_reveal"] += 1
    if visible_marks or key_components or material_features:
        scores["identity_material_confirm"] += 2
    if can_handheld and has_any(scene_words):
        scores["scene_fit_showcase"] += 1

    priority = [
        "feature_operation_demo",
        "premium_texture_reveal",
        "scene_fit_showcase",
        "problem_solution_pair",
        "usage_result_demo",
        "aspirational_lifestyle_result",
        "identity_material_confirm",
        "direct_benefit_proof",
    ]
    expression_strategy = max(priority, key=lambda strategy: (scores[strategy], -priority.index(strategy)))

    primary_value = _commerce_pick_specific_value(
        proof_terms,
        _commerce_strategy_keywords(expression_strategy),
        fallback_terms=selling_points + functional_features + material_features,
        fallback="核心卖点看得见",
    )
    remaining_proof_terms = [
        term for term in proof_terms
        if _clean_short_sentence(term, max_chars=18) != primary_value
    ]
    result_value = _commerce_pick_specific_value(
        remaining_proof_terms,
        result_words + scene_words + lifestyle_words,
        fallback_terms=[
            point for point in selling_points + functional_features + material_features
            if _clean_short_sentence(point, max_chars=18) != primary_value
        ],
        fallback=primary_value,
    )
    detail_value = _commerce_pick_specific_value(
        material_features + visible_marks + key_components + selling_points,
        material_words + operation_words,
        fallback_terms=proof_terms,
        fallback=primary_value,
    )
    scene_context = _commerce_scene_phrase(usage_scene, target_audience)
    component = _commerce_pick_component(key_components, operation_words)
    operation_action = _commerce_pick_operation_action(allowed_actions, component)

    source_action = f"商品保持在素材首帧场景中清楚可见，只呈现一个能证明「{primary_value}」的状态、结构或使用关系。"
    result_action = "人物或道具只做围绕结果状态的轻微辅助动作，画面重点保持在商品和已成立的使用关系上。"
    result_state = f"商品与{scene_context}中的日常道具形成明确使用关系，画面能证明「{result_value}」。"

    if expression_strategy == "premium_texture_reveal":
        result_state = f"商品处在{scene_context}的近景位置，材质、纹理、包装或标识区域清楚可见，画面证明「{primary_value}」。"
    elif expression_strategy == "feature_operation_demo":
        result_state = f"{component}与周边使用道具关系清楚，商品处于已完成操作的状态，画面证明「{primary_value}」。"
    elif expression_strategy == "scene_fit_showcase":
        result_state = f"商品与{scene_context}形成清楚比例、收纳、搭配或使用关系，画面证明「{primary_value}」。"
    elif expression_strategy == "problem_solution_pair":
        result_state = f"商品处在问题被改善后的结果画面里，周边道具更整洁、顺手或舒适，画面证明「{result_value}」。"
    elif expression_strategy == "usage_result_demo":
        result_state = f"商品已经在真实使用结果状态中，人物和日常道具关系自然，画面证明「{result_value}」。"
    elif expression_strategy == "aspirational_lifestyle_result":
        result_state = f"商品位于有仪式感的{scene_context}里，和道具、光线、人物动作共同证明「{result_value}」。"
    elif expression_strategy == "identity_material_confirm":
        result_state = f"商品关键结构、材质和可见标识保持一致，处在真实使用环境里，画面证明「{detail_value}」。"
    elif expression_strategy == "direct_benefit_proof":
        result_state = f"商品处在真实使用结果中，画面通过位置、接触关系和周边道具证明「{primary_value}」。"

    return {
        "expression_strategy": expression_strategy,
        "primary_value": primary_value,
        "result_value": result_value,
        "confirm_caption": detail_value,
        "action_caption": primary_value,
        "result_caption": result_value,
        "source_action": source_action,
        "result_action": result_action,
        "result_state": result_state,
    }


def _commerce_subtitles(proof_plan: dict[str, str]) -> list[str]:
    primary_value = str(proof_plan.get("primary_value") or "核心卖点").strip()
    return [
        _safe_user_caption(
            proof_plan.get("confirm_caption", ""),
            fallback=primary_value,
            max_chars=DEFAULT_SUBTITLE_MAX_CHARS,
        ),
        _safe_user_caption(
            proof_plan.get("action_caption", ""),
            fallback=primary_value,
            max_chars=DEFAULT_SUBTITLE_MAX_CHARS,
        ),
        _safe_user_caption(
            proof_plan.get("result_caption", ""),
            fallback=str(proof_plan.get("result_value") or "日常使用更方便"),
            max_chars=DEFAULT_SUBTITLE_MAX_CHARS,
        ),
    ]


def _commerce_reconstruction_risk(identity_card: dict[str, Any], anchor: dict[str, Any]) -> str:
    text = " ".join(
        _string_list(identity_card.get("visible_marks"))
        + _string_list(identity_card.get("key_components"))
        + [str(identity_card.get("product_type", "")), str(anchor.get("visual_role", ""))]
    ).lower()
    if any(word in text for word in ("logo", "标识", "商标", "屏幕", "键盘", "铰链", "笔记本", "电脑", "laptop")):
        return "high"
    if any(word in text for word in ("文字", "标签", "包装", "复杂")):
        return "medium"
    return "low"


def _commerce_prompt_source_confirm(
    *,
    product_type: str,
    appearance: str,
    style: str,
    scenario: dict[str, str],
) -> str:
    fallback = (
        f"这是 5 秒图生视频。第一帧就是上传素材中的同一件{product_type}，先不要改变地点。"
        f"0-1 秒完全保持首帧构图，让观众看清真实商品；1-5 秒只让{scenario['source_place']}里的自然光轻微变化，"
        "镜头有非常轻的生活感呼吸，但商品不跳动、不自己移动。"
        f"商品外观必须保持：{appearance}。"
        "画面里可以有桌面、手边普通道具和柔和阴影，但不要新增第二个同类商品，"
        "不要新增非商品自带文字、UI、水印或字幕；商品自带 logo、标识或字样只保持首帧已有外观，不要改写。"
        f"整体风格：{style}。这个镜头只负责确认真实商品身份，不讲复杂剧情。"
    )
    return _render_prompt_skill_template(
        "commerce_scene.source_confirm",
        {
            "product_type": product_type,
            "appearance": appearance,
            "source_place": scenario.get("source_place", "素材原始场景"),
            "style": style,
        },
        fallback,
    )


def _commerce_prompt_material_action(
    *,
    product_type: str,
    appearance: str,
    style: str,
    proof_plan: dict[str, str],
    first_point: str,
) -> str:
    fallback = (
        f"这是 5 秒图生视频，第一帧仍然来自上传素材中的同一件{product_type}。"
        "本镜头只发生一个动作，不换地点，不跳到新场景。"
        f"动作分段：0-1 秒保持商品稳定；1-4 秒{proof_plan['source_action']}；4-5 秒动作结束并停稳。"
        f"这个动作要让观众看出卖点「{first_point}」，不要只靠字幕。"
        f"商品外观必须保持：{appearance}。"
        "商品自带 logo、标识或字样只保持首帧已有外观，不要新增、改写或重画。"
        "手部必须先接触商品再移动，接触点和支撑关系清楚；商品不能悬浮、跳动、变形、自己走动或凭空换角度。"
        "背景只允许在首帧素材场景基础上轻微扩展，例如桌面边缘、背包一角或自然光，不要跨到户外、办公室入口等新地点。"
        f"整体风格：{style}。"
    )
    return _render_prompt_skill_template(
        "commerce_scene.material_action_proof",
        {
            "product_type": product_type,
            "appearance": appearance,
            "source_action": proof_plan.get("source_action", "让商品在素材场景里完成一个低风险动作并停稳"),
            "first_point": first_point,
            "style": style,
        },
        fallback,
    )


def _commerce_prompt_new_scene_result(
    *,
    product_type: str,
    appearance: str,
    style: str,
    scenario: dict[str, str],
    proof_plan: dict[str, str],
    second_point: str,
    risk_level: str,
) -> str:
    identity_clause = (
        "如果标识或细节无法稳定复刻，宁可用距离、角度或手部轻微遮挡弱化可读标识，也不要生成错误 logo 或随机文字。"
        if risk_level == "high"
        else "商品颜色、轮廓和关键结构要和前面素材理解一致，可以自然融入新场景。"
    )
    fallback = (
        f"这是 5 秒文生视频，是硬切后的新镜头，不承接上一镜的时间、地点或背景。"
        f"镜头开始时已经在{scenario['result_place']}，不要表现从上一场景走过来的过程。"
        f"画面主体是{scenario['human']}和同一件{product_type}的使用结果状态。"
        f"商品外观参考：{appearance}。{identity_clause}"
        f"画面具体状态：{proof_plan['result_state']}。"
        f"动作只做结果展示：0-1 秒建立新场景；1-4 秒{proof_plan['result_action']}；4-5 秒停在清楚的商品结果画面。"
        f"这一镜要用画面证明「{second_point}」，不是只出现一个泛生活场景。"
        "不要新增非商品自带文字、UI、购物按钮、水印；不要出现第二个同类主商品；不要把商品改成其他品类。"
        f"整体风格：{style}。"
    )
    return _render_prompt_skill_template(
        "commerce_scene.new_scene_result",
        {
            "product_type": product_type,
            "appearance": appearance,
            "style": style,
            "result_place": scenario.get("result_place", "真实使用场景"),
            "human": scenario.get("human", "普通用户的手部或半身动作"),
            "result_state": proof_plan.get("result_state", "商品处在真实使用位置，关系清楚"),
            "result_action": proof_plan.get("result_action", "人物只做轻微辅助动作，让画面结果证明卖点"),
            "second_point": second_point,
            "identity_clause": identity_clause,
        },
        fallback,
    )


def _v3_opening_source_scene_action_template(
    product_action: dict[str, Any],
    identity_card: dict[str, Any],
) -> dict[str, Any]:
    """Build a source-scene identity shot without product-specific choreography."""

    motion_affordance = _safe_dict(identity_card.get("motion_affordance"))
    allowed_actions = _string_list(motion_affordance.get("allowed_actions"))
    unsafe_words = ("喝", "嘴", "翻转", "旋转", "开合", "打开", "跨面", "倒", "拆", "走路", "飞", "跳")

    base_forbidden = _unique_list(
        product_action.get("forbidden", [])
        + ["换地点", "跨场景", "多步骤动作", "未接触先移动"]
    )
    base_review_focus = _unique_list(
        product_action.get("review_focus", [])
        + ["素材场景延展", "真实接触关系", "单一低风险动作"]
    )
    template = {
        **product_action,
        "visual": (
            "第一帧承接上传素材中的真实商品和原有承托环境，沿同一地点自然延展，"
            "不换地点、不新增第二个商品"
        ),
        "camera_motion": "定镜或极轻微推进，保持素材场景和商品身份稳定",
        "scene_elements": ["上传素材原场景", "商品", "手部或自然光"],
        "forbidden": base_forbidden,
        "review_focus": base_review_focus,
    }

    safe_allowed = [
        action for action in allowed_actions
        if action and not any(word in action for word in unsafe_words)
    ]
    if safe_allowed:
        template["action"] = (
            "开场镜只确认真实商品身份，不主动设计使用动作；"
            "如果素材中已有自然接触或承托关系，只保持关系稳定并允许轻微光影变化。"
        )
        template["selected_opening_action"] = "identity_confirm_only"
        return template

    template["action"] = str(product_action.get("action", "")).strip() or (
        "商品保持在素材原位置稳定展示，只允许自然光影和很小幅度的镜头推进，"
        "不移动、不换地点。"
    )
    template["selected_opening_action"] = "template_safe_fallback"
    return template


_SENTENCE_BOUNDARY_RE = re.compile(r"([。！？!?；;])")


def _selling_point_phrases(raw_points: Any, *, max_count: int = 6) -> list[str]:
    """把用户输入的逗号串卖点拆成可逐镜绑定的短语。"""

    if isinstance(raw_points, str):
        raw_items = [raw_points]
    elif isinstance(raw_points, list):
        raw_items = raw_points
    else:
        raw_items = []

    phrases: list[str] = []
    for item in raw_items:
        text = str(item or "").strip()
        if not text:
            continue
        for part in re.split(r"[,，、/；;|｜\n]+", text):
            phrase = re.sub(r"\s+", "", part).strip("。！？!? ")
            if phrase and phrase not in phrases:
                phrases.append(phrase)
            if len(phrases) >= max_count:
                return phrases
    return phrases


def _clean_short_sentence(text: str, *, max_chars: int = 24) -> str:
    """清理字幕/口播短句，优先保留完整句子，避免逗号硬截断。"""

    normalized = re.sub(r"\s+", "", str(text or ""))
    normalized = re.sub(r"[，,、；;。！？!?]+$", "", normalized)
    if not normalized:
        return ""

    parts = [part for part in re.split(r"[，,、；;。！？!?\n]+", normalized) if part]
    deduped: list[str] = []
    for part in parts:
        if part and part not in deduped:
            deduped.append(part)
    normalized = "，".join(deduped) if deduped else normalized
    if len(normalized) <= max_chars:
        return normalized

    sentence_end = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(normalized):
        if match.end() <= max_chars:
            sentence_end = match.end()
        else:
            break
    if sentence_end > 0:
        return normalized[:sentence_end].rstrip("，,、；;")

    return normalized[:max_chars].rstrip("，,、；;")


def _product_context_title(product_context: dict[str, Any]) -> str:
    identity_card = product_context.get("product_identity_card", {})
    if not isinstance(identity_card, dict):
        identity_card = {}
    return _clean_short_sentence(
        product_context.get("product_title")
        or product_context.get("title")
        or identity_card.get("product_type")
        or product_context.get("product_type")
        or "",
        max_chars=12,
    )


def _product_context_selling_points(product_context: dict[str, Any]) -> list[str]:
    identity_card = product_context.get("product_identity_card", {})
    if not isinstance(identity_card, dict):
        identity_card = {}
    raw_points: list[Any] = []
    for key in ("selling_points", "functional_features", "material_features"):
        value = product_context.get(key)
        if isinstance(value, list):
            raw_points.extend(value)
        elif value:
            raw_points.append(value)
    for key in ("functional_features", "material_features", "key_components", "visible_marks"):
        value = identity_card.get(key)
        if isinstance(value, list):
            raw_points.extend(value)
        elif value:
            raw_points.append(value)
    return _selling_point_phrases(raw_points)


def _fallback_public_caption(
    product_context: dict[str, Any],
    role: str,
    *,
    scene_goal: str = "",
    max_chars: int = DEFAULT_SUBTITLE_MAX_CHARS,
) -> str:
    """为降级路径生成用户可见短字幕，不使用固定 CTA、内部策略名或痛点模板。"""

    title = _product_context_title(product_context)
    selling_points = _product_context_selling_points(product_context)
    structured = product_context.get("structured_requirements") or {}
    if not isinstance(structured, dict):
        structured = {}
    usage_scene = _clean_short_sentence(
        product_context.get("usage_scene") or structured.get("usage_scene", ""),
        max_chars=max_chars,
    )
    scene_goal_short = _clean_short_sentence(scene_goal, max_chars=max_chars)
    first_point = selling_points[0] if selling_points else ""
    second_point = selling_points[1] if len(selling_points) > 1 else ""
    third_point = selling_points[2] if len(selling_points) > 2 else ""
    role_key = str(role or "").strip().lower()

    candidates_by_role = {
        "hook": [first_point, usage_scene, title, scene_goal_short, "使用场景看清"],
        "problem": [first_point, usage_scene, title, scene_goal_short, "使用场景看清"],
        "context": [first_point, usage_scene, title, scene_goal_short, "使用场景看清"],
        "product_reveal": [title, first_point, scene_goal_short, "外观细节看清"],
        "product_confirm": [title, first_point, scene_goal_short, "外观细节看清"],
        "product_hero": [title, first_point, scene_goal_short, "外观细节看清"],
        "feature_demo": [first_point, second_point, scene_goal_short, title, "核心卖点看清"],
        "detail_proof": [second_point, third_point, first_point, scene_goal_short, "细节看得见"],
        "commerce_action_proof": [first_point, second_point, scene_goal_short, "动作结果看清"],
        "commerce_result": [third_point, second_point, first_point, usage_scene, scene_goal_short, "卖点结果看得见"],
        "commerce_result_scene": [third_point, second_point, first_point, usage_scene, scene_goal_short, "卖点结果看得见"],
        "usage_or_lifestyle": [third_point, second_point, first_point, usage_scene, scene_goal_short, "使用结果看得见"],
        "lifestyle_result": [third_point, usage_scene, second_point, first_point, "使用结果看得见"],
        "cta": [third_point, second_point, first_point, usage_scene, scene_goal_short, "卖点结果看得见"],
        "value_close": [third_point, second_point, first_point, usage_scene, scene_goal_short, "卖点结果看得见"],
    }
    candidates = candidates_by_role.get(role_key, [first_point, second_point, title, scene_goal_short, "卖点看得见"])
    for candidate in candidates:
        cleaned = _clean_short_sentence(candidate, max_chars=max_chars)
        if cleaned and not _is_internal_or_generic_caption(cleaned):
            return cleaned
    return _clean_short_sentence("卖点看得见", max_chars=max_chars)


def _asset_profile_map(asset_analysis: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(profile.get("asset_id", "")).strip(): profile
        for profile in asset_analysis.get("asset_profiles", [])
        if isinstance(profile, dict) and str(profile.get("asset_id", "")).strip()
    }


def _merge_asset_profile(asset: dict[str, Any] | None, profiles_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not asset:
        return {}
    profile = profiles_by_id.get(str(asset.get("asset_id", "")).strip(), {})
    return {**profile, **asset}


def _v3_asset_usage_metadata(
    asset: dict[str, Any] | None,
    profile: dict[str, Any] | None,
    *,
    role: str,
    material_strategy: str,
    prompt_skill: str,
) -> dict[str, Any]:
    if not asset:
        return {}
    visual_role = str((profile or {}).get("visual_role") or asset.get("visual_role") or "").strip()
    suitable_for = (profile or {}).get("suitable_for") or asset.get("suitable_for") or []
    reason = str((profile or {}).get("reason") or "").strip()
    if not reason:
        if visual_role in {"appearance_anchor", "full_product_anchor"}:
            reason = "素材画像标记为完整外观锚点，适合建立商品身份。"
        elif visual_role in {"detail_reference", "logo_detail", "brand_detail"}:
            reason = "素材画像标记为细节参考，适合卖点或细节镜头。"
        else:
            reason = "素材可用作当前商品镜头参考。"
    return {
        "selected_asset_ids": [str(asset.get("asset_id", "")).strip()],
        "visual_role": visual_role,
        "suitable_for": suitable_for,
        "is_identity_critical": role in {"product_reveal", "product_hero", "cta"},
        "material_strategy": material_strategy,
        "selected_prompt_skill": prompt_skill,
        "asset_usage_reason": reason,
    }


def _v3_select_role_asset(
    allocation: dict[str, list[str]],
    assets_by_id: dict[str, dict[str, Any]],
    *,
    role: str,
    fallback: dict[str, Any],
) -> dict[str, Any]:
    if role == "detail_reference":
        candidate_ids = allocation.get("detail_reference", [])
    else:
        candidate_ids = allocation.get("appearance_anchor", [])
    candidates = [assets_by_id[asset_id] for asset_id in candidate_ids if asset_id in assets_by_id]
    return _best_asset_with_geometric_fallback(candidates) or fallback


def _v3_product_shot(
    *,
    shot_index: int,
    role: str,
    duration: int,
    subtitle: str,
    scene_goal: str,
    action_template: dict[str, Any],
    product_type: str,
    identity_card: dict[str, Any],
    asset_id: str,
) -> dict[str, Any]:
    appearance = str(identity_card.get("appearance_summary", "")).strip() or f"上传素材中的同一件{product_type}"
    must_preserve = _string_list(identity_card.get("must_preserve", []))
    forbidden_changes = _string_list(identity_card.get("forbidden_changes", []))
    forbidden_variation = _unique_list(
        forbidden_changes
        + action_template.get("forbidden", [])
        + [
            "不得变成其他商品",
            "不得新增第二个同类商品",
            "不得改变颜色、轮廓、结构或品牌标识",
            "不得新增非商品自带文字、字符、按钮或 UI",
        ]
    )
    visual_description = (
        f"{action_template['visual']}。商品身份：{appearance}。"
        "商品必须来自上传素材首帧，保持同一件商品，不重绘为类似商品。"
    )
    subtitle_text = _safe_user_caption(subtitle, fallback=scene_goal, max_chars=DEFAULT_SUBTITLE_MAX_CHARS)
    shot = {
        "shot_index": shot_index,
        "duration_seconds": duration,
        "narrative_role": role,
        "scene_goal": scene_goal,
        "purpose": scene_goal,
        "initial_state": "首帧是上传素材中的真实商品，商品主体完整清楚。",
        "action": str(action_template["action"]),
        "final_state": "动作结束后商品仍保持真实素材中的身份、结构、颜色、标识和主体数量。",
        "shot_type": str(action_template.get("shot_type", "近景")),
        "camera_motion": str(action_template.get("camera_motion", "定镜")),
        "subject_appearance": appearance,
        "subject_position": str(action_template.get("subject_position", "商品位于画面主体区域")),
        "acting_direction": str(action_template["action"]),
        "scene_elements": action_template.get("scene_elements", []),
        "visual_description": visual_description,
        "subtitle": subtitle_text,
        "voiceover": subtitle_text,
        "asset_id": asset_id,
        "render_strategy": "image_to_video",
        "transition_type": "hard_cut",
        "continuity_group": "",
        "anchor_last_frame": False,
        "product_presence": "required",
        "identity_strictness": "high",
        "allowed_variation": action_template.get("allowed", ["自然光影轻微变化"]),
        "forbidden_variation": forbidden_variation,
        "review_focus": _unique_list(
            action_template.get("review_focus", [])
            + ["商品身份", "标识一致性", "结构一致性", "动作是否完成"]
        ),
        "product_identity_constraints": must_preserve,
        "product_identity_card": identity_card,
        "risk_notes": [],
        "video_prompt_constraints": {
            "must_preserve": must_preserve,
            "must_avoid": forbidden_variation,
        },
    }
    shot["video_prompt"] = _compose_final_video_prompt(shot, product_type=product_type, appearance=appearance)
    shot["force_video_prompt"] = True
    shot["final_prompt_source"] = "strategy_contract"
    return shot


def _v3_product_result_shot(
    *,
    shot_index: int,
    role: str,
    duration: int,
    subtitle: str,
    scene_goal: str,
    action_template: dict[str, Any],
    product_type: str,
    identity_card: dict[str, Any],
    asset_id: str,
) -> dict[str, Any]:
    """商品结果镜：A 保守策略也必须有商品内容，不能把带货结果写成无商品铺垫。"""

    shot = _v3_product_shot(
        shot_index=shot_index,
        role=role,
        duration=duration,
        subtitle=subtitle,
        scene_goal=scene_goal,
        action_template=action_template,
        product_type=product_type,
        identity_card=identity_card,
        asset_id=asset_id,
    )
    appearance = str(identity_card.get("appearance_summary", "")).strip() or f"上传素材中的同一件{product_type}"
    result_visual = str(action_template.get("visual", "")).strip()
    result_action = str(action_template.get("action", "")).strip()
    fallback_visual_description = (
        f"{result_visual}。本镜是带货结果证明镜，不是无商品铺垫镜头。"
        f"画面必须一直保留同一件{product_type}作为主体，商品身份：{appearance}。"
        "可以在首帧素材场景边缘轻微扩展生活道具，但不要跨到完全不同地点。"
    )
    visual_description = _render_prompt_skill_template(
        "source_scene_extension.product_result_scene",
        {
            "product_type": product_type,
            "appearance": appearance,
            "result_visual": result_visual,
            "result_action": result_action,
            "scene_goal": scene_goal,
        },
        fallback_visual_description,
    )
    shot.update(
        {
            "initial_state": "首帧仍使用上传素材中的真实商品，商品主体清楚可见，硬切但不更换成自由生成商品。",
            "visual_description": visual_description,
            "action": result_action,
            "final_state": "动作结束时商品仍清楚可见，观众能从商品和周边道具关系理解卖点结果。",
            "asset_requirement": "必须使用上传商品图作为首帧锚点；结果镜也必须保留商品主体，不允许生成纯生活空镜。",
            "identity_strictness": "medium",
            "review_focus": _unique_list(
                _string_list(shot.get("review_focus"))
                + ["卖点是否由商品画面证明", "是否错误变成无商品空镜"]
            ),
            "risk_notes": _unique_list(
                _string_list(shot.get("risk_notes"))
                + ["A 保守策略的结果镜仍要求商品出现；避免回退到 product_presence=forbidden 的泛生活镜头。"]
            ),
        }
    )
    shot["video_prompt"] = _compose_final_video_prompt(shot, product_type=product_type, appearance=appearance)
    shot["force_video_prompt"] = True
    shot["final_prompt_source"] = "strategy_contract"
    return shot


def _compose_final_video_prompt(
    shot: dict[str, Any],
    *,
    product_type: str,
    appearance: str,
) -> str:
    """把上游策略合同压成给视频模型的唯一自然语言 prompt。"""

    render_strategy = str(shot.get("render_strategy", "")).strip()
    duration = int(shot.get("duration_seconds", 5) or 5)
    visual = str(shot.get("visual_description", "")).strip()
    action = str(shot.get("action", "")).strip()
    initial_state = str(shot.get("initial_state", "")).strip()
    final_state = str(shot.get("final_state", "")).strip()
    camera = str(shot.get("camera_motion", "")).strip() or "定镜或非常轻微的生活感手持运动"
    avoid_items = _string_list(shot.get("forbidden_variation", []))
    avoid_text = "；".join(avoid_items[:6])
    if render_strategy == "image_to_video":
        opening = (
            f"这是 {duration} 秒图生视频，竖屏 9:16。第一帧就是上传素材中的同一件{product_type}，"
            "后续只在这个首帧基础上自然延展，不重新生成类似商品。"
        )
    else:
        opening = (
            f"这是 {duration} 秒文生视频，竖屏 9:16。这个镜头是独立的新场景，"
            "只执行本镜头描述的一个画面任务。"
        )
    parts = [
        opening,
        f"镜头开始：{initial_state}" if initial_state else "",
        f"场景与构图：{visual}" if visual else "",
        f"唯一主要动作：{action}" if action else "",
        f"镜头结束：{final_state}" if final_state else "",
        f"商品外观：{appearance}",
        f"镜头运动：{camera}，不要用快速运动掩盖商品身份。",
        (
            "商品自带 logo、标识或文字只保持首帧已有外观；如果无法稳定复刻，宁可弱化可读性，"
            "不要生成错误 logo、随机字母或额外文字。"
        ),
        f"必须避免：{avoid_text}" if avoid_text else "",
        "整体风格：真实写实的商业短视频，柔和自然光，背景克制干净，单镜头单动作。",
    ]
    prompt = "\n".join(part for part in parts if part)
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)
    return prompt.strip()


def _v3_context_shot(
    *,
    shot_index: int,
    role: str,
    duration: int,
    subtitle: str,
    scene_goal: str,
    action_template: dict[str, Any],
    product_type: str,
) -> dict[str, Any]:
    forbidden_items = _context_forbidden_product_terms(product_type)
    forbidden_variation = _unique_list(
        forbidden_items
        + [
            "不得出现待售商品",
            "不得出现同类商品",
            "不得出现品牌 logo、品牌文字或可读商品标签",
            "不得把商品素材画进场景",
            "不得出现画面内文字、字符、按钮或 UI",
        ]
    )
    visual = str(action_template.get("context_visual") or action_template.get("visual", "")).strip()
    action = str(action_template.get("context_action") or action_template.get("action", "")).strip()
    camera_motion = str(action_template.get("context_camera_motion") or action_template.get("camera_motion", "定镜")).strip()
    scene_elements = action_template.get("context_scene_elements") or action_template.get("scene_elements", [])
    subtitle_text = _clean_short_sentence(subtitle)
    return {
        "shot_index": shot_index,
        "duration_seconds": duration,
        "narrative_role": role,
        "scene_goal": scene_goal,
        "purpose": scene_goal,
        "initial_state": "真实生活场景开场，画面中没有待售商品或同类商品。",
        "action": action,
        "final_state": "场景完成痛点铺垫，仍不出现待售商品或同类商品。",
        "shot_type": str(action_template.get("context_shot_type", "中景")),
        "camera_motion": camera_motion,
        "subject_appearance": "真实人物或生活环境，不展示待售商品。",
        "subject_position": str(action_template.get("context_subject_position", "人物或环境位于画面主体区域")),
        "acting_direction": action,
        "scene_elements": scene_elements,
        "visual_description": (
            f"{visual}。这是无商品剧情铺垫镜头，用来表达使用场景和痛点，并为下一镜的真实商品做铺垫。"
            f"避免清晰展示{product_type}、同类商品、logo、品牌文字或可读标签；"
            f"但可以保留不带品牌、看不清细节的使用情景线索（如对应的收纳包、桌面摆放位置、使用环境），"
            f"让这一镜和下一镜的真实商品自然衔接。"
        ),
        "subtitle": subtitle_text,
        "voiceover": subtitle_text,
        "asset_id": "",
        "render_strategy": "text_to_video",
        "transition_type": "hard_cut",
        "continuity_group": "",
        "anchor_last_frame": False,
        "product_presence": "forbidden",
        "identity_strictness": "low",
        "allowed_variation": ["人物动作自然", "环境光影自然变化", "背景道具可轻微运动但不能像商品"],
        "forbidden_variation": forbidden_variation,
        "review_focus": ["剧情场景是否成立", "是否没有出现同类商品", "是否没有品牌文字或 logo"],
        "product_identity_constraints": [],
        "product_identity_card": {},
        "risk_notes": ["无商品场景只承担叙事，不承担商品外观展示。"],
        "video_prompt_constraints": {
            "must_preserve": ["真实生活场景", "自然动作", "商业短视频质感"],
            "must_avoid": forbidden_variation,
        },
    }


def _product_fidelity_template_source(product_type: str) -> str:
    return "product_fidelity_v3_skill_guided"


def _product_fidelity_action_templates(product_type: str) -> dict[str, dict[str, Any]]:
    common_forbidden = ["翻转", "旋转", "悬浮", "飞入", "变形", "复杂手部交互"]
    return {
        "hook": {
            "visual": f"素材首帧中的同一件{product_type}完整清楚地出现在画面主体区域",
            "action": "商品保持稳定，只允许自然光影或镜头呼吸发生轻微变化。",
            "camera_motion": "定镜",
            "shot_type": "近景",
            "scene_elements": ["素材首帧场景", product_type],
            "forbidden": common_forbidden,
            "review_focus": ["商品完整外观"],
        },
        "feature": {
            "visual": f"同一件{product_type}保持完整外观，围绕卖点形成可见结构、材质、容量、尺寸或使用关系证据",
            "action": "商品保持在素材首帧场景中清楚可见，只展示一个能被画面证明的商品状态或使用关系。",
            "camera_motion": "定镜",
            "shot_type": "近景",
            "scene_elements": ["素材首帧场景", product_type, "必要的真实接触或承托关系"],
            "forbidden": common_forbidden,
            "review_focus": ["卖点是否由画面证明", "商品结构", "动作是否有意义"],
        },
        "scene": {
            "visual": f"同一件{product_type}在真实使用关系中稳定展示，周围道具用于证明结果而不是抢主体",
            "action": "人物或道具只做围绕结果状态的轻微辅助动作，画面重点证明商品和使用关系已经成立。",
            "camera_motion": "定镜或极轻微横移",
            "shot_type": "中近景",
            "scene_elements": ["真实使用关系", product_type],
            "forbidden": common_forbidden,
            "review_focus": ["商品主体清楚", "结果关系清楚", "环境不新增同类商品"],
        },
    }


def _detail_action_template(product_type: str, identity_card: dict[str, Any]) -> dict[str, Any]:
    focus = "、".join(_string_list(identity_card.get("visible_marks", [])) or _string_list(identity_card.get("must_preserve", []))[:2])
    focus_text = focus or "关键结构和材质"
    return {
        "visual": f"同一件{product_type}的{focus_text}细节清楚可见，使用真实素材局部构图",
        "action": "商品保持完全稳定，只允许轻微光线变化突出材质和结构。",
        "camera_motion": "定镜，不推近，不旋转",
        "shot_type": "特写",
        "scene_elements": [product_type, "真实细节"],
        "forbidden": ["结构变形", "标识重绘", "文字字符", "推近导致变形"],
        "review_focus": ["细节一致性", "标识形状", "材质结构"],
    }


def _cta_action_template(product_type: str) -> dict[str, Any]:
    return {
        "visual": f"同一件{product_type}回到完整稳定展示，背景干净，画面预留本地字幕空间",
        "action": "商品稳定定格，只允许轻微自然光影变化，完整收束。",
        "camera_motion": "定镜",
        "shot_type": "近景",
        "scene_elements": [product_type, "干净背景"],
        "forbidden": ["新增同类商品", "商品变形", "logo 发光", "文字字符", "复杂动作"],
        "review_focus": ["完整商品", "收尾稳定", "CTA 空间"],
    }


def _default_selling_point(product_type: str) -> str:
    if _is_laptop_product_type(product_type):
        return "轻薄机身"
    if _is_cup_product_type(product_type):
        return "日常随手用"
    return "真实外观"


def _context_problem_subtitle(product_type: str, first_point: str) -> str:
    if _is_laptop_product_type(product_type):
        return "出门办公更省心"
    if _is_cup_product_type(product_type):
        return "出门前随手带上"
    return first_point


def _context_problem_goal(product_type: str, first_point: str) -> str:
    if _is_laptop_product_type(product_type):
        return "用无商品办公/通勤场景铺垫需求，下一镜再回到真实笔记本展示卖点。"
    if _is_cup_product_type(product_type):
        return "用无商品出门/办公场景铺垫随手补水需求，下一镜再回到真实水杯展示卖点。"
    return f"用无商品生活场景铺垫需求：{first_point}，下一镜再回到真实商品。"


def _context_forbidden_product_terms(product_type: str) -> list[str]:
    if _is_laptop_product_type(product_type):
        return ["笔记本", "电脑", "键盘", "屏幕", "平板", "laptop", "notebook"]
    if _is_cup_product_type(product_type):
        return ["水杯", "杯子", "保温杯", "水壶", "瓶子", "饮料瓶", "cup", "bottle"]
    return [product_type, "同类商品", "商品包装"]


def _detail_subtitle(product_type: str, identity_card: dict[str, Any]) -> str:
    if _is_laptop_product_type(product_type):
        return "细节一眼识别"
    if _is_cup_product_type(product_type):
        return "杯盖杯身看清"
    marks = _string_list(identity_card.get("visible_marks", []))
    return f"{marks[0]}看得见" if marks else "细节看得见"


def _scene_subtitle(product_type: str, first_point: str) -> str:
    if _is_laptop_product_type(product_type):
        return "通勤收纳更轻松"
    if _is_cup_product_type(product_type):
        return "放在手边刚好"
    return first_point


def _is_laptop_product_type(product_type: str) -> bool:
    normalized = product_type.lower()
    return any(keyword in normalized for keyword in ("笔记本", "电脑", "laptop", "notebook"))


def _is_cup_product_type(product_type: str) -> bool:
    normalized = product_type.lower()
    return any(keyword in normalized for keyword in ("水杯", "杯", "保温杯", "cup", "bottle"))


def _build_template_script_plan_stub(
    storyboard: list[dict[str, Any]],
    product_type: str,
    source: str,
    duration: int = 18,
    product_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    subtitle_list = [s.get("subtitle", s.get("seedance_prompt", "")[:20]) for s in storyboard]
    timeline = []
    cursor = 0
    for shot in storyboard:
        shot_duration = int(shot.get("duration_seconds", 0) or 0)
        timeline.append(
            {
                "start_seconds": cursor,
                "end_seconds": cursor + shot_duration,
                "role": shot.get("narrative_role", ""),
                "subtitle": shot.get("subtitle", ""),
                "visual_intent": (shot.get("visual_description") or shot.get("seedance_prompt", ""))[:80],
                "scene_before": shot.get("initial_state", ""),
                "action": shot.get("action", ""),
                "scene_after": shot.get("final_state", ""),
                "visible_entities": shot.get("scene_elements", []),
                "physical_constraints": shot.get("forbidden_variation", []),
                "asset_requirements": [shot.get("asset_requirement", "")] if shot.get("asset_requirement") else [],
                "continuity_mode": "continue_previous" if shot.get("continuity_group") else "new_scene",
                "transition_reason": "硬切到新的商品证明镜头" if shot.get("transition_type") == "hard_cut" else "",
                "shot_type": shot.get("shot_type", ""),
                "camera_movement": shot.get("camera_motion", ""),
                "scene_description": shot.get("visual_description", ""),
                "subject_appearance": shot.get("subject_appearance", ""),
                "subject_position": shot.get("subject_position", ""),
                "acting_direction": shot.get("acting_direction") or shot.get("action", ""),
                "dialogue": "[No Dialogue]",
                "scene_elements": shot.get("scene_elements", []),
                "cut_reason": "当前镜头动作或结果状态完成后切换",
            }
        )
        cursor += shot_duration
    role_arc = " -> ".join(
        str(shot.get("narrative_role", "")).strip() for shot in storyboard if str(shot.get("narrative_role", "")).strip()
    )
    semantic_coverage = _semantic_coverage_from_storyboard(storyboard)
    plan_contract = _build_plan_contract(
        strategy_id=_strategy_id_from_source(source),
        strategy_family=_strategy_family_from_source(source),
        storyboard=storyboard,
        semantic_coverage=semantic_coverage,
    )
    full_subtitle_script = "，".join(str(item).strip() for item in subtitle_list if str(item).strip())
    rich_story_text = _rich_story_text_from_product_context(
        storyboard,
        product_type,
        product_context or {},
    ) or _rich_story_text_from_storyboard(storyboard, product_type)
    return {
        "grounded_product_type": product_type,
        "narrative_arc": role_arc or "product_hook -> feature_demo -> detail_proof -> cta",
        "context_reconstruction": _context_reconstruction_from_storyboard(storyboard, product_type),
        "rich_story_text": rich_story_text,
        "core_message": subtitle_list[0] if subtitle_list else "",
        "user_emotion": "先确认真实商品，再看到卖点证明和使用结果。",
        "key_visual_moments": [
            str(shot.get("scene_goal") or shot.get("visual_description") or "").strip()
            for shot in storyboard
            if str(shot.get("scene_goal") or shot.get("visual_description") or "").strip()
        ],
        "full_subtitle_script": full_subtitle_script,
        "subtitle_script": full_subtitle_script,
        "voiceover_script": full_subtitle_script,
        "hook": subtitle_list[0] if subtitle_list else "",
        "body": subtitle_list[1:4],
        "cta": subtitle_list[-1] if subtitle_list else "",
        "target_duration_seconds": duration,
        "beats": timeline,
        "tone": "真实写实的商品带货短视频",
        "style_notes": "素材优先、单镜头单动作、硬切剪辑感、商品身份优先。",
        "semantic_coverage": semantic_coverage,
        "plan_contract": plan_contract,
        "_source": source,
    }


_SEMANTIC_ROLE_HINTS: dict[str, set[str]] = {
    "hook": {"attention"},
    "problem": {"attention"},
    "context": {"attention"},
    "product_reveal": {"attention", "identity"},
    "product_hero": {"attention", "identity"},
    "product_confirm": {"attention", "identity"},
    "feature_demo": {"proof"},
    "detail_proof": {"proof"},
    "commerce_action_proof": {"proof"},
    "commerce_result": {"result", "conversion_intent"},
    "commerce_result_scene": {"result", "conversion_intent"},
    "lifestyle_result": {"result"},
    "cta": {"conversion_intent"},
}
_SEMANTIC_COVERAGE_ORDER = ["attention", "identity", "proof", "result", "conversion_intent"]

_INTERNAL_OR_FALLBACK_CAPTIONS = {
    "真实外观确认",
    "看看这个好物",
    "看看这个",
    "点击了解更多",
    "点击查看详情",
    "立即查看详情",
    "真实体验",
}


def _is_internal_or_generic_caption(text: str) -> bool:
    normalized = re.sub(r"\s+", "", str(text or ""))
    if not normalized:
        return True
    if normalized in _INTERNAL_OR_FALLBACK_CAPTIONS:
        return True
    return any(
        phrase in normalized
        for phrase in (
            "真实外观确认",
            "看看这个好物",
            "看看这个",
            "点击了解更多",
            "点击查看详情",
            "点击下方",
            "立即查看详情",
            "真实体验",
            "先看看这个问题",
            "谁还没遇到过这个问题",
        )
    )


def _safe_user_caption(raw_text: str, *, fallback: str, max_chars: int = DEFAULT_SUBTITLE_MAX_CHARS) -> str:
    candidate = _clean_short_sentence(raw_text, max_chars=max_chars)
    if _is_internal_or_generic_caption(candidate):
        candidate = _clean_short_sentence(fallback, max_chars=max_chars)
    if _is_internal_or_generic_caption(candidate):
        candidate = "卖点看得见"
    return candidate


def _strategy_family_from_source(source: str) -> str:
    source_text = str(source or "")
    if "product_fidelity_v3" in source_text:
        return "template_product_fidelity"
    if "B_ideal_commerce_scene" in source_text:
        return "ideal_commerce_scene"
    if "template_path_b" in source_text:
        return "template_path_b"
    return "legacy"


def _strategy_id_from_source(source: str) -> str:
    source_text = str(source or "")
    if "product_fidelity_v3" in source_text:
        return "product_fidelity_v3"
    if "B_ideal_commerce_scene" in source_text:
        return "B_ideal_commerce_scene"
    if "template_path_b" in source_text:
        return "template_path_b"
    return source_text or "legacy"


def _plan_strategy_family(script_plan: dict[str, Any], storyboard: list[dict[str, Any]]) -> str:
    contract = _safe_dict(script_plan.get("plan_contract"))
    family = str(contract.get("strategy_family", "")).strip()
    if family:
        return family
    source = str(script_plan.get("_source") or "").strip()
    if source:
        return _strategy_family_from_source(source)
    for shot in storyboard:
        source = str(shot.get("planner_source") or "").strip()
        if source:
            family = _strategy_family_from_source(source)
            if family != "legacy":
                return family
    return "legacy"


def _semantic_coverage_from_storyboard(storyboard: list[dict[str, Any]]) -> list[str]:
    coverage: set[str] = set()
    for shot in storyboard:
        role = str(shot.get("narrative_role", "")).strip()
        coverage.update(_SEMANTIC_ROLE_HINTS.get(role, set()))
        product_presence = str(shot.get("product_presence", "")).strip().lower()
        render_strategy = str(shot.get("render_strategy", "")).strip()
        text = " ".join(
            str(shot.get(key, ""))
            for key in ("purpose", "scene_goal", "visual_description", "action", "final_state", "subtitle")
        )
        if product_presence == "required" or render_strategy == "image_to_video":
            coverage.add("identity")
        if any(word in text for word in ("卖点", "证明", "展示", "细节", "性能", "容量", "轻薄", "便携", "收纳")):
            coverage.add("proof")
        if any(word in text for word in ("结果", "使用", "通勤", "办公", "背包", "手边", "场景")):
            coverage.add("result")
        if any(word in text for word in ("点击", "了解", "购买", "下单", "收束", "带货")):
            coverage.add("conversion_intent")
    return [item for item in _SEMANTIC_COVERAGE_ORDER if item in coverage]


def _ordered_semantic_items(items: set[str]) -> list[str]:
    return [item for item in _SEMANTIC_COVERAGE_ORDER if item in items]


def _build_plan_contract(
    *,
    strategy_id: str,
    strategy_family: str,
    storyboard: list[dict[str, Any]],
    semantic_coverage: list[str],
) -> dict[str, Any]:
    return {
        "strategy_id": strategy_id,
        "strategy_family": strategy_family,
        "expected_shape": {
            "contract_type": "semantic_coverage",
            "shot_count": len(storyboard),
        },
        "required_coverage": ["attention", "identity", "proof", "result", "conversion_intent"],
        "semantic_coverage": semantic_coverage,
        "role_mapping": {
            str(shot.get("narrative_role", "")).strip(): _ordered_semantic_items(
                _SEMANTIC_ROLE_HINTS.get(str(shot.get("narrative_role", "")).strip(), set())
            )
            for shot in storyboard
            if str(shot.get("narrative_role", "")).strip()
        },
        "subtitle_policy": "用户可见字幕必须来自卖点/场景表达，不得使用内部诊断标签。",
        "prompt_policy": "视频模型只接收上游合成后的单一自然语言 prompt，结构字段只用于审核和日志。",
        "fallback_policy": "非 legacy 策略审核失败时不静默降级到旧保守模板。",
    }


def _attach_strategy_contract_to_storyboard(
    storyboard: list[dict[str, Any]],
    *,
    strategy_id: str,
    strategy_family: str,
) -> None:
    semantic_coverage = _semantic_coverage_from_storyboard(storyboard)
    plan_contract = _build_plan_contract(
        strategy_id=strategy_id,
        strategy_family=strategy_family,
        storyboard=storyboard,
        semantic_coverage=semantic_coverage,
    )
    for shot in storyboard:
        shot["semantic_coverage"] = semantic_coverage
        shot["plan_contract"] = plan_contract


def _context_reconstruction_from_storyboard(storyboard: list[dict[str, Any]], product_type: str) -> dict[str, Any]:
    goals = [str(shot.get("scene_goal", "")).strip() for shot in storyboard if str(shot.get("scene_goal", "")).strip()]
    actions = [str(shot.get("action", "")).strip() for shot in storyboard if str(shot.get("action", "")).strip()]
    return {
        "scene_setting": f"围绕真实素材中的{product_type}进行商品身份确认、卖点证明和结果收束。",
        "plot_development": " -> ".join(goals) if goals else "真实商品展示 -> 卖点证明 -> 结果收束",
        "emotional_tendency": "从确认真实商品，到理解卖点，再到产生继续了解的兴趣。",
        "speaking_intent": "用短字幕配合画面证明商品价值，不使用内部诊断语言。",
        "causal_chain": actions or goals,
    }


def _rich_story_text_from_storyboard(storyboard: list[dict[str, Any]], product_type: str) -> str:
    parts = []
    for shot in storyboard:
        role = str(shot.get("narrative_role", "")).strip()
        goal = str(shot.get("scene_goal", "")).strip()
        action = str(shot.get("action", "")).strip()
        final_state = str(shot.get("final_state", "")).strip()
        text = "，".join(item for item in (role, goal, action, final_state) if item)
        if text:
            parts.append(text)
    return f"这条视频围绕真实素材中的{product_type}展开。" + "；".join(parts)


def _rich_story_text_from_product_context(
    storyboard: list[dict[str, Any]],
    product_type: str,
    product_context: dict[str, Any],
) -> str:
    title = str(
        product_context.get("product_title")
        or product_context.get("title")
        or product_type
        or "这款商品"
    ).strip()
    usage_scene = str(product_context.get("usage_scene") or "").strip()
    audience = str(product_context.get("target_audience") or product_context.get("audience") or "").strip()
    selling_points = _string_list(product_context.get("selling_points", []))
    concrete_goals = [
        str(shot.get("scene_goal") or shot.get("visual_description") or "").strip()
        for shot in storyboard
        if str(shot.get("scene_goal") or shot.get("visual_description") or "").strip()
        and str(shot.get("scene_goal") or shot.get("visual_description") or "").strip()
        not in {"使用情境", "核心卖点", "卖点细节", "结果收束"}
    ]
    concrete_actions = [
        str(shot.get("action") or "").strip()
        for shot in storyboard
        if str(shot.get("action") or "").strip()
        and str(shot.get("action") or "").strip() not in {"展示商品", "展示卖点"}
        and not _story_action_is_camera_only(str(shot.get("action") or ""))
    ]

    subject = audience or "目标用户"
    scene = f"在{usage_scene}" if usage_scene else "在真实使用场景中"
    proof = "、".join(selling_points[:3]) if selling_points else f"{product_type}的核心卖点"
    if concrete_goals:
        development = "，随后".join(concrete_goals[:3])
    elif concrete_actions:
        development = "，随后".join(concrete_actions[:3])
    else:
        development = f"先建立{scene}的需求，再用画面证明{proof}"
    return (
        f"这条视频讲述{subject}{scene}遇到需求后，看到并使用{title}，"
        f"通过{development}，证明{proof}，最后形成继续了解或购买的理由。"
    )


def _story_action_is_camera_only(action: str) -> bool:
    normalized = re.sub(r"\s+", "", str(action or ""))
    if not normalized:
        return True
    camera_terms = ("推近", "拉远", "横移", "环绕", "定镜", "定格", "固定镜头", "轻微推近", "镜头")
    return any(term in normalized for term in camera_terms) and not any(
        term in normalized
        for term in ("放进", "拿出", "打开", "收纳", "倒水", "喝水", "行走", "使用", "对比", "展示")
    )


def plan_storyboard_from_template(
    product_context: dict[str, Any],
    asset_analysis: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    路径B：模板+1次LLM，取代 plan_script + plan_director_storyboard 三跳流水线。

    品牌信息（visible_marks/appearance_summary/texture_notes）直接作为模板变量注入，
    不经过任何LLM中转摘要，消除信息递减问题。

    返回 (storyboard, script_plan_stub)，script_plan_stub 是兼容后续流程的最小剧本结构。
    """
    identity_card = product_context.get("product_identity_card", {})
    brand = identity_card.get("brand_name", "")
    product_type = identity_card.get("product_type", "商品")
    appearance = identity_card.get("appearance_summary", "")
    marks = identity_card.get("visible_marks", [])
    marks_text = "、".join(str(m) for m in marks) if marks else "品牌标识"
    texture = identity_card.get("texture_notes", "材质表面")
    style = product_context.get("user_style", "高质感电商产品视频")
    selling_points = product_context.get("selling_points", [])

    usage_scene = product_context.get("usage_scene", "") or ""
    target_audience = (
        product_context.get("target_audience", "")
        or (product_context.get("structured_requirements") or {}).get("target_audience", "")
        or "普通用户"
    )
    sp1 = selling_points[0] if selling_points else f"{product_type}的核心价值"
    sp2 = selling_points[1] if len(selling_points) > 1 else sp1
    selling_points_text = "、".join(selling_points[:3]) if selling_points else f"{product_type}优质体验"

    # 根据目标受众和使用场景生成具体的使用场景描述
    _usage_scene_map = {
        "游戏": f"游戏玩家在电脑桌前激烈操作，或外出携带{product_type}进入网咖/宿舍的场景",
        "办公": f"上班族在办公桌前或咖啡馆里使用{product_type}处理工作的场景",
        "学生": f"学生在宿舍或图书馆使用{product_type}学习的场景",
        "通勤": f"通勤者在地铁或公交上使用{product_type}的场景",
        "户外": f"用户在户外、公园或旅途中使用{product_type}的场景",
        "运动": f"运动/健身场景中使用{product_type}的场景",
        "家居": f"用户在家里客厅或卧室日常使用{product_type}的场景",
    }
    usage_scene_desc = ""
    for kw, desc in _usage_scene_map.items():
        if kw in usage_scene or kw in target_audience:
            usage_scene_desc = desc
            break
    if not usage_scene_desc:
        usage_scene_desc = f"{target_audience}在日常生活中使用{product_type}的真实场景，环境自然真实"

    best_anchor = _find_best_appearance_anchor(asset_analysis)
    has_real_anchor = bool(best_anchor and best_anchor.get("file_path"))
    if has_real_anchor:
        storyboard = _plan_product_fidelity_v3_storyboard(product_context, asset_analysis, best_anchor)
        product_type = str(identity_card.get("product_type", "商品")).strip() or "商品"
        source = _product_fidelity_template_source(product_type)
        script_plan_stub = _build_template_script_plan_stub(
            storyboard,
            product_type=product_type,
            source=source,
            duration=sum(int(shot.get("duration_seconds", 0) or 0) for shot in storyboard),
            product_context=product_context,
        )
        _flow_print(
            "[plan_storyboard_from_template] 商品保真 V3 完成："
            f"{len(storyboard)}个分镜，source={source}"
        )
        return storyboard, script_plan_stub

    script_plan = plan_script(product_context)
    storyboard = plan_director_storyboard(product_context, script_plan, asset_analysis)
    if not storyboard:
        _flow_print(
            "[plan_storyboard_from_template] no usable product anchor; "
            "director planner unavailable, using conservative fallback"
        )
        script_plan = _fallback_conservative_script(product_context, {})
        storyboard = _fallback_conservative_storyboard(product_context, script_plan)

    storyboard = _ensure_storyboard_continuity(_normalize_storyboard(storyboard))
    duration = sum(int(shot.get("duration_seconds", 0) or 0) for shot in storyboard)
    if duration <= 0:
        duration = int(product_context.get("duration_seconds", 15) or 15)
    product_type = str(
        product_context.get("product_type")
        or identity_card.get("product_type")
        or "product"
    ).strip() or "product"
    script_plan_stub = _build_template_script_plan_stub(
        storyboard,
        product_type=product_type,
        source="template_path_b_no_anchor_director",
        duration=duration,
        product_context=product_context,
    )
    if script_plan.get("hook"):
        script_plan_stub["hook"] = script_plan.get("hook", "")
    if script_plan.get("body"):
        script_plan_stub["body"] = script_plan.get("body", [])
    if script_plan.get("cta"):
        script_plan_stub["cta"] = script_plan.get("cta", "")

    _flow_print(
        "[plan_storyboard_from_template] no-anchor path completed via director planner: "
        f"{len(storyboard)} shots"
    )
    return storyboard, script_plan_stub


def _find_best_appearance_anchor(asset_analysis: dict[str, Any]) -> dict[str, Any] | None:
    """从素材分析中找最适合 detail_proof 镜头的商品外观锚点素材。"""

    assets = [
        asset
        for asset in asset_analysis.get("assets", [])
        if isinstance(asset, dict)
        and asset.get("asset_type") == "image"
        and asset.get("is_supported")
        and (asset.get("file_path") or asset.get("anchor_file_path"))
    ]
    profiles = [profile for profile in asset_analysis.get("asset_profiles", []) if isinstance(profile, dict)]
    assets_by_id = {str(asset.get("asset_id", "")): asset for asset in assets}
    merged_candidates: list[dict[str, Any]] = []
    for profile in profiles:
        asset = assets_by_id.get(str(profile.get("asset_id", "")))
        if asset:
            merged_candidates.append({**profile, **asset})
    merged_candidates.extend(asset for asset in assets if asset not in merged_candidates)

    for role in ("appearance_anchor", "full_product_anchor"):
        candidates = [
            item
            for item in merged_candidates
            if str(item.get("visual_role", "")).strip() == role
            and (item.get("file_path") or item.get("anchor_file_path"))
        ]
        if candidates:
            return _best_asset_with_geometric_fallback(candidates)
    candidate_anchors = [
        item
        for item in merged_candidates
        if _is_full_product_anchor(item) and (item.get("file_path") or item.get("anchor_file_path"))
    ]
    if candidate_anchors:
        return _best_asset_with_geometric_fallback(candidate_anchors)
    for role in ("logo_detail", "brand_detail", "detail"):
        candidates = [
            item
            for item in merged_candidates
            if str(item.get("visual_role", "")).strip() == role
            and (item.get("file_path") or item.get("anchor_file_path"))
        ]
        if candidates:
            return _best_asset_with_geometric_fallback(candidates)
    if merged_candidates:
        return _best_asset_with_geometric_fallback(merged_candidates)
    return None


def plan_script(
    product_context: dict[str, Any],
    director_decision: dict[str, Any],
    previous_issues: list[str] | None = None,
) -> dict[str, Any]:
    """生成带货视频剧本规划。"""

    print("[video_generation_workflow] 开始剧本规划。", flush=True)
    duration = max(3, int(product_context.get("duration_seconds", 15)))
    identity_card = _safe_dict(product_context.get("product_identity_card", {}))
    expected_product_type = (
        str(identity_card.get("product_type", "")).strip()
        or str(product_context.get("product_title", "")).strip()
    )
    prompt = {
        "task": "为电商商品生成 15 秒带货短视频剧本。先理解商品和策略，再创作。",
        "product_grounding_contract": {
            "expected_product_title": product_context.get("product_title", ""),
            "expected_product_type": expected_product_type,
            "appearance_summary": identity_card.get("appearance_summary", ""),
            "must_preserve": identity_card.get("must_preserve", []),
            "forbidden_changes": identity_card.get("forbidden_changes", []),
            "hard_rule": (
                "整条剧本只能围绕 expected_product_type 创作，不得改写成其他商品。"
                "商品类别、外观、动作和字幕必须与商品身份卡一致；无法确认的事实不要编造。"
            ),
        },
        "creative_brief": {
            "hook_principle": "前 2 秒决定用户是否划走。hook 必须直接、有冲击力、让用户产生「这跟我有关」的感觉",
            "body_principle": "中段每一个卖点对应一个画面。不堆砌参数，说用户能 get 到的利益点",
            "cta_principle": "结尾用商品价值、结果状态或下一步意图收束。不要固定写“点击查看/下单”套话。",
        },
        "narrative_workflow": (
            "第一阶段：Context Reconstruction。先重建场景设定、情节推进、情绪走向、表达意图和因果链。"
            "第二阶段：Shot-Level Semantic Planning。再写 rich_story_text，并拆分为可独立拍摄的 beats。"
            "每个 beat 必须写清场景前态、单一动作、场景后态、镜头类型、运镜、主体位置、对白和切镜原因。"
            "第三阶段：Multi-Round Adaptive Error Correction。输出前逐项检查对白完整性、主体外观一致性、场景连贯性和位置物理合理性。"
        ),
        "script_generation_stages": {
            "context_reconstruction": "把零散输入融合为明确的场景、情节、情绪、表达意图和因果链。",
            "shot_level_semantic_planning": (
                "只在明确的镜头语言、场景或叙事变化处切镜。每个 beat 必须自洽、可单独拍摄、可交给视频模型执行。"
            ),
            "adaptive_error_correction": "根据 previous_issues 修复脚本，不要原样重复上一版。",
        },
        "script_verification_modules": {
            "dialogue_completeness": "每个 beat 必须明确写出对白或 [No Dialogue]。",
            "subject_appearance_consistency": "跟踪主体外观，不能在连续镜头中无原因改变。",
            "scene_coherence": "跟踪环境元素，换场必须有叙事理由。",
            "positional_physical_rationality": "主体位置、道具状态、动作和镜头几何必须符合真实物理关系。",
        },
        "instruction": "只返回 JSON，不要返回解释文字。",
        "examples": [
            {
                "scenario": "蓝牙耳机，场景氛围风",
                "narrative_arc": "hook -> feature_demo -> detail_proof -> cta",
                "context_reconstruction": {
                    "scene_setting": "通勤地铁车厢，环境嘈杂。",
                    "plot_development": "先建立噪声困扰，再展示耳机带来的专注体验。",
                    "emotional_tendency": "从烦躁转为放松。",
                    "speaking_intent": "用具体通勤场景表达降噪价值。",
                    "causal_chain": ["地铁噪声造成困扰", "用户佩戴耳机", "用户恢复专注"],
                },
                "full_subtitle_script": "戴上一秒，世界安静了。ANC 主动降噪，地铁上也能沉浸听歌。单耳 4.2g，戴一天耳朵不疼。通勤路上更专注。",
                "voiceover_script": "戴上它，世界瞬间安静。主动降噪让你在地铁也能沉浸听歌，单耳只有 4.2 克，戴一整天都不累，通勤路上更专注。",
                "beats": [
                    {
                        "start_seconds": 0,
                        "end_seconds": 3,
                        "role": "hook",
                        "message": "用嘈杂地铁建立降噪需求",
                        "subtitle": "通勤噪声太吵？",
                        "visual_intent": "地铁车厢内，用户被环境噪声打扰。",
                        "scene_before": "用户站在嘈杂地铁车厢内。",
                        "action": "用户抬手触碰耳机，环境氛围从嘈杂转为专注。",
                        "scene_after": "用户神情放松，继续站在同一车厢。",
                        "visible_entities": ["用户", "耳机", "地铁车厢"],
                        "physical_constraints": ["用户动作自然", "耳机外观保持稳定"],
                        "asset_requirements": ["耳机佩戴参考或不突出品牌的通勤场景"],
                        "continuity_mode": "new_scene",
                        "transition_reason": "用通勤噪声建立使用情境。",
                        "shot_type": "中景",
                        "camera_movement": "固定镜头",
                        "scene_description": "真实地铁车厢，用户站在扶手旁。",
                        "subject_appearance": "通勤用户佩戴真实耳机，服饰和耳机外观稳定。",
                        "subject_position": "用户位于画面中央，扶手位于画面右侧。",
                        "acting_direction": "用户抬手触碰耳机后自然放松。",
                        "dialogue": "[No Dialogue]",
                        "scene_elements": ["地铁车厢", "扶手", "通勤用户", "耳机"],
                        "cut_reason": "用户情绪稳定后切到商品细节证明。"
                    },
                ],
                "closing_intent": "让用户记住「通勤路上更专注」这个结果。",
                "hook": "戴上一秒，世界安静了。",
                "body": [
                    "ANC 主动降噪，地铁上也能沉浸听歌",
                    "单耳 4.2g，戴一天耳朵不疼",
                    "充电 10 分钟，听歌 2 小时"
                ],
                "cta": "通勤路上更专注。",
                "style_notes": "温柔知性，不吵闹，像朋友推荐",
            }
        ],
        "output_format": {
            "grounded_product_type": "必须原样填写 product_grounding_contract.expected_product_type，用于审核剧本没有偏题",
            "product_grounding_summary": "说明剧本如何围绕真实商品展开，不要引入其他商品主体",
            "narrative_arc": "叙事弧线结构，格式为 hook -> feature -> proof -> cta",
            "context_reconstruction": {
                "scene_setting": "整体场景设定和环境元素",
                "plot_development": "情节如何逐步推进",
                "emotional_tendency": "用户感受如何变化",
                "speaking_intent": "字幕、对白或口播想表达什么",
                "causal_chain": ["显式因果链，说明前后画面为什么衔接"],
            },
            "story_title": "剧本标题",
            "rich_story_text": "丰富故事文本，写清楚场景、商品出现方式、画面细节、卖点证明和结尾收束，供导演后续取舍",
            "core_message": "这条视频最想让用户记住的一句话",
            "user_emotion": "希望用户产生的感受或认知变化",
            "key_visual_moments": ["关键画面时刻列表，必须可拍且最好能对应上传素材"],
            "full_subtitle_script": "完整 15 秒字幕文案，一段连贯的短视频文案",
            "subtitle_script": "同 full_subtitle_script，兼容更自然命名",
            "voiceover_script": "完整口播文案",
            "beats": [
                {
                    "start_seconds": "节拍开始秒数",
                    "end_seconds": "节拍结束秒数",
                    "role": "叙事角色：hook / feature_demo / detail_proof / cta",
                    "message": "这个节拍要表达什么",
                    "subtitle": "这个节拍对应的字幕",
                    "visual_intent": "这个节拍希望看到的画面",
                    "evidence_refs": ["该节拍依赖的用户卖点、商品身份或素材角色"],
                    "scene_before": "镜头开始前，画面中可见的场景、主体和状态",
                    "action": "这个节拍中发生的单一核心动作或变化，必须可物理实现",
                    "scene_after": "动作完成后，画面中可见的场景、主体和状态",
                    "visible_entities": ["画面中允许出现的具体实体"],
                    "physical_constraints": ["该节拍必须遵守的物理或商品身份约束"],
                    "asset_requirements": ["该节拍需要哪类真实素材；不需要时也要明确说明"],
                    "continuity_mode": "new_scene / continue_previous",
                    "transition_reason": "切换到新场景时说明叙事原因；延续上一场景时可以留空",
                    "shot_type": "特写 / 近景 / 中景 / 全景",
                    "camera_movement": "固定镜头 / 轻微平移 / 合理的镜头运动",
                    "scene_description": "可直接用于拍摄的场景描述",
                    "subject_appearance": "当前镜头中主体外观；连续镜头必须保持一致",
                    "subject_position": "主体和关键道具在画面中的位置关系",
                    "acting_direction": "人物、商品或道具的具体表演和动作指导",
                    "dialogue": "明确对白；没有对白时必须写 [No Dialogue]",
                    "scene_elements": ["需要跨镜头跟踪的环境元素和道具"],
                    "cut_reason": "为什么在这里结束当前镜头并切换"
                }
            ],
            "closing_intent": "结尾要让用户记住什么或做什么",
            "hook": "开头 1-3 秒吸引用户停留的钩子文案，一句话",
            "body": ["中段卖点展开，每条一个用户利益点，不超过 15 字"],
            "cta": "结尾引导文案，有行动感",
            "style_notes": "整条视频的语气、节奏、风格约束",
            "target_duration_seconds": duration,
        },
        "asset_capability_plan": product_context.get("asset_capability_plan", {}),
        "product_context": _product_context_for_llm(product_context),
        "director_decision": director_decision,
        "creative_direction": _format_creative_direction(director_decision),
        "structured_requirements": product_context.get("structured_requirements", {}),
        "narrative_rules": {
            "subtitle_rule": "字幕应该先作为完整文案生成，再切分到各个分镜，而不是每个分镜独立想一句",
            "closing_rule": "必须有明确收尾或 CTA，不能突然中断",
            "fabrication_rule": "不能编造用户未提供且素材无法证明的品牌、型号、参数或功能",
            "story_detail_rule": "rich_story_text 不能只是摘要，必须包含足够画面细节，后续导演可以从中删减，但不能没有内容可拆。",
            "executable_beat_rule": "每个 beat 必须是可执行合同：scene_before -> action -> scene_after，并列出 visible_entities、physical_constraints、asset_requirements。",
            "continuity_rule": "场景延续时 continuity_mode 使用 continue_previous；切换到新场景时使用 new_scene，并填写 transition_reason，避免无原因跳切。",
            "asset_capability_rule": "必须优先使用 asset_capability_plan.supported_shot_types。asset_capability_plan.unsupported_shot_types 中的动作不能写进 beat.action，只能通过字幕/旁白表达。",
        },
        "diversity_hint": "避免使用「你还在为XX烦恼吗」或「今天给大家推荐一款」之类的套路句式。从具体的使用场景或用户情绪切入。",
        "previous_issues": previous_issues or [],
    }
    llm_result = _call_text_llm(prompt, purpose="script_plan", temperature=0.85)

    # 优先使用 LLM 返回的结构化剧本。
    # 如果模型没有返回可解析 JSON，就用规则兜底，避免工作流直接中断。
    if llm_result["ok"]:
        parsed_script = _extract_json_from_text(llm_result["content"])
        script = _normalize_script_plan(parsed_script, product_context)
        if script:
            script["llm_enabled"] = True
            script["llm_notes"] = llm_result["content"]
            print("[video_generation_workflow] 剧本规划完成：llm_enabled=True", flush=True)
            return script
        print("[video_generation_workflow] 剧本规划失败：LLM 返回内容无法解析。", flush=True)
        return {
            "narrative_arc": "",
            "story_title": "",
            "rich_story_text": "",
            "core_message": "",
            "user_emotion": "",
            "key_visual_moments": [],
            "full_subtitle_script": "",
            "subtitle_script": "",
            "voiceover_script": "",
            "beats": [],
            "closing_intent": "",
            "hook": "",
            "body": [],
            "cta": "",
            "tone": _merge_style_text(product_context),
            "style_notes": _merge_style_text(product_context),
            "target_duration_seconds": duration,
            "llm_enabled": True,
            "llm_notes": llm_result["content"],
            "llm_error": "LLM 返回内容无法解析成剧本 JSON。",
        }
    else:
        notes = "当前未启用 LLM，使用规则生成的基础剧本。"

    selling_points = _product_context_selling_points(product_context)
    first_point = selling_points[0] if selling_points else _fallback_public_caption(product_context, "feature_demo", scene_goal="核心卖点看清")
    second_point = selling_points[1] if len(selling_points) > 1 else _fallback_public_caption(product_context, "detail_proof", scene_goal=first_point)
    result_point = selling_points[2] if len(selling_points) > 2 else _fallback_public_caption(product_context, "cta", scene_goal=first_point)
    title = _product_context_title(product_context) or "这款商品"
    identity_card = product_context.get("product_identity_card", {})
    product_summary = str(identity_card.get("appearance_summary", "")).strip() or str(product_context.get("asset_summary", "")).strip()
    rich_story_text = (
        f"视频先从用户最关心的「{first_point}」切入，画面要让 {title} 的商品主体尽早出现。"
        f"如果素材能证明商品外观，就围绕这些真实素材展开：{product_summary or '展示商品真实外观、细节和使用场景'}。"
        "中段用一到两个镜头把卖点具体化，而不是只堆叠口号；结尾回到商品完整画面，用结果状态完成商业表达。"
    )
    script = {
        "narrative_arc": "value_hook -> feature_demo -> detail_proof -> cta",
        "context_reconstruction": {
            "scene_setting": "真实使用环境切入，随后切换到简洁稳定的商品展示环境。",
            "plot_development": f"先建立与「{first_point}」相关的使用情境，再展示真实商品主体和细节，最后用可见结果收束。",
            "emotional_tendency": "从产生注意，转为理解商品价值，最后形成继续了解的兴趣。",
            "speaking_intent": f"让用户记住 {title} 的核心优势是「{first_point}」。",
            "causal_chain": ["建立使用情境", "商品自然出现", "细节证明卖点", "结果画面收束"],
        },
        "story_title": f"{title}的{first_point}卖点短视频",
        "rich_story_text": rich_story_text,
        "core_message": f"{title}的核心优势是{first_point}",
        "user_emotion": "让用户在短时间内理解商品卖点，并产生继续了解的兴趣。",
        "key_visual_moments": [
            "商品主体尽早出现，建立真实感。",
            f"围绕「{first_point}」展示一个可见的商品细节或使用场景。",
            "结尾回到商品完整画面并给出明确行动引导。",
        ],
        "full_subtitle_script": f"{title}，{first_point}，值得拥有。",
        "subtitle_script": f"{title}，{first_point}，值得拥有。",
        "voiceover_script": "",
        "beats": [
            {
                "start_seconds": 0,
                "end_seconds": 3,
                "role": "hook",
                "message": "从真实使用情境切入，吸引注意。",
                "subtitle": _fallback_public_caption(product_context, "hook", scene_goal=first_point),
                "visual_intent": "使用环境或人物动作先建立与卖点相关的场景，不急于堆叠商品参数。",
                "scene_before": "用户处于与商品卖点有关的真实使用环境中。",
                "action": "用户完成一个能体现使用情境的简单动作。",
                "scene_after": "画面保留场景关系，并为商品自然出现留出空间。",
                "visible_entities": ["用户", "真实使用环境"],
                "physical_constraints": ["人物动作自然", "商品尚未出现时不得凭空生成同类商品"],
                "asset_requirements": ["不需要真实商品素材；使用不突出商品品牌的环境铺垫"],
                "continuity_mode": "new_scene",
                "transition_reason": "用真实使用情境建立用户注意力。",
            },
            {
                "start_seconds": 3,
                "end_seconds": duration - 5,
                "role": "feature_demo",
                "message": f"用真实商品画面证明核心卖点：{first_point}",
                "subtitle": first_point,
                "visual_intent": "商品主体清晰出现，通过稳定画面呈现最重要的用户利益点。",
                "scene_before": "使用情境已经建立，画面准备切换到商品主体。",
                "action": "真实商品主体稳定出现，并保持完整外观。",
                "scene_after": "商品主体完整可见，用户能够识别核心外观。",
                "visible_entities": ["真实商品主体", "简洁展示环境"],
                "physical_constraints": ["商品外观保持稳定", "不得改变品牌、颜色、结构或关键部件"],
                "asset_requirements": ["必须使用真实商品外观锚点素材"],
                "continuity_mode": "new_scene",
                "transition_reason": "从使用情境切换到商品价值证明。",
            },
            {
                "start_seconds": duration - 5,
                "end_seconds": duration - 2,
                "role": "detail_proof",
                "message": "使用真实细节增强可信度。",
                "subtitle": _fallback_public_caption(product_context, "detail_proof", scene_goal=second_point),
                "visual_intent": "固定构图展示材质、结构或品牌识别区域，不让商品发生不合理变化。",
                "scene_before": "商品主体已经完整展示。",
                "action": "镜头稳定切换到一个能够证明卖点的真实细节。",
                "scene_after": "商品细节清晰可见，外观仍与上传素材一致。",
                "visible_entities": ["真实商品主体", "商品细节"],
                "physical_constraints": ["商品外观保持稳定", "细节镜头不得引入不合理旋转、折叠或变形"],
                "asset_requirements": ["优先使用真实商品细节素材；没有时复用外观锚点"],
                "continuity_mode": "continue_previous",
                "transition_reason": "",
            },
            {
                "start_seconds": duration - 2,
                "end_seconds": duration,
                "role": "cta",
                "message": "结果画面收束。",
                "subtitle": _fallback_public_caption(product_context, "value_close", scene_goal=result_point),
                "visual_intent": "回到商品完整画面，画面稳定收束并为本地字幕预留空间。",
                "scene_before": "用户已经看到商品主体和关键细节。",
                "action": "镜头回到稳定的商品完整画面，并保留能表达卖点结果的构图。",
                "scene_after": "商品全景稳定保持，视频完整收束。",
                "visible_entities": ["真实商品主体", "简洁展示环境"],
                "physical_constraints": ["商品外观保持稳定", "结尾不得突然中断或引入新动作"],
                "asset_requirements": ["必须使用真实商品外观锚点素材"],
                "continuity_mode": "continue_previous",
                "transition_reason": "",
            },
        ],
        "closing_intent": f"用稳定结果画面强化「{result_point}」。",
        "hook": f"先用一句话突出：{product_context.get('product_title', '这款商品')} 的 {first_point}。",
        "body": [
            f"展示商品外观，并强调 {point}。"
            for point in selling_points[:3]
        ],
        "cta": _fallback_public_caption(product_context, "value_close", scene_goal=result_point),
        "tone": _merge_style_text(product_context),
        "style_notes": _merge_style_text(product_context),
        "target_duration_seconds": duration,
        "script_contract_version": "v3_paper_style_cinematic_script",
        "expected_product_type": expected_product_type,
        "grounded_product_type": expected_product_type,
        "product_grounding_summary": f"规则兜底剧本仅围绕 {expected_product_type or '当前商品'} 展开。",
        "llm_enabled": llm_result["ok"],
        "llm_notes": notes,
    }
    script["beats"] = _enrich_fallback_script_beats(script["beats"])
    print(f"[video_generation_workflow] 剧本规划完成：llm_enabled={llm_result['ok']}", flush=True)
    return script


def _enrich_fallback_script_beats(beats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给规则兜底剧本补齐论文风格的可执行镜头字段。"""

    paper_fields_by_role = {
        "hook": {
            "shot_type": "中景",
            "camera_movement": "固定镜头",
            "scene_description": "真实生活环境，人物完成一个简单动作，画面克制写实。",
            "subject_appearance": "人物和普通无品牌道具保持真实稳定，待售商品尚未出现。",
            "subject_position": "人物位于画面中部，普通道具位于使用空间两侧。",
            "acting_direction": "人物自然整理使用空间，为后续商品出现留出位置。",
            "dialogue": "[No Dialogue]",
            "scene_elements": ["真实使用环境", "人物", "普通无品牌道具"],
            "cut_reason": "用户问题已经建立，切入商品解决方案。",
        },
        "feature_demo": {
            "shot_type": "中景",
            "camera_movement": "固定镜头",
            "scene_description": "简洁稳定的商品展示环境，真实商品主体完整可见。",
            "subject_appearance": "商品外观、颜色、结构和标识区域与上传素材保持一致。",
            "subject_position": "商品主体位于画面中央，背景保持简洁。",
            "acting_direction": "商品稳定出现，只允许轻微自然光影变化。",
            "dialogue": "[No Dialogue]",
            "scene_elements": ["真实商品主体", "简洁展示环境"],
            "cut_reason": "商品主体已经建立，切入可信细节证明。",
        },
        "detail_proof": {
            "shot_type": "近景",
            "camera_movement": "固定镜头",
            "scene_description": "保持同一商品和展示环境，稳定呈现一个真实细节。",
            "subject_appearance": "商品关键细节与上传素材保持一致。",
            "subject_position": "商品细节位于画面中央，主体仍可被识别。",
            "acting_direction": "镜头保持稳定，不引入结构变化。",
            "dialogue": "[No Dialogue]",
            "scene_elements": ["真实商品主体", "商品细节", "简洁展示环境"],
            "cut_reason": "卖点证明完成，切入结尾收束。",
        },
        "cta": {
            "shot_type": "中景",
            "camera_movement": "固定镜头",
            "scene_description": "回到商品完整画面，为 CTA 字幕留出空间。",
            "subject_appearance": "商品完整外观与上传素材保持一致。",
            "subject_position": "商品主体位于画面中央，字幕区域保持干净。",
            "acting_direction": "画面稳定保持，不增加新的动作或道具。",
            "dialogue": "[No Dialogue]",
            "scene_elements": ["真实商品主体", "简洁展示环境", "字幕留白区域"],
            "cut_reason": "视频在完整商品画面上自然结束。",
        },
    }
    return [
        {**beat, **paper_fields_by_role.get(str(beat.get("role", "")), paper_fields_by_role["feature_demo"])}
        for beat in beats
    ]



def _fallback_narrative_role(index: int, shot_count: int) -> str:
    """兜底分镜的叙事角色，保证没有 LLM 时也不是纯静态展示。"""

    if index == 0:
        return "hook"
    if index == shot_count - 1:
        return "cta"
    return "feature_demo"


def _fallback_shot_action(index: int, shot_count: int) -> str:
    """兜底分镜动作，给视频模型最基本的状态变化目标。"""

    if index == 0:
        return "镜头从干净背景缓慢推近商品主体，建立注意力。"
    if index == shot_count - 1:
        return "镜头保持商品主体稳定，画面节奏放缓并留出字幕空间。"
    return "镜头围绕商品轻微移动，突出一个明确卖点或细节。"


def _canonical_product_type_for_review(product_type: Any) -> str:
    """归一化常见商品类型别名，避免模型使用中英文别名时被误判为偏题。"""

    normalized = str(product_type or "").strip().lower()
    aliases = {
        "笔记本电脑": ("笔记本电脑", "游戏本", "轻薄本", "laptop", "gaming notebook"),
        "保温杯": ("保温杯", "thermos", "vacuum cup", "insulated cup"),
        "耳机": ("耳机", "headphone", "earphone", "earbuds"),
        "手机": ("手机", "smartphone", "mobile phone"),
    }
    for canonical_type, words in aliases.items():
        if any(word in normalized for word in words):
            return canonical_type
    return normalized


def review_script_plan(script_plan: dict[str, Any]) -> dict[str, Any]:
    """审核剧本结构是否足够支撑后续分镜生成。"""

    issues: list[str] = []
    hook = str(script_plan.get("hook", "")).strip()
    body = script_plan.get("body", [])
    cta = str(script_plan.get("cta", "")).strip()
    tone = str(script_plan.get("tone", "")).strip()
    rich_story_text = str(script_plan.get("rich_story_text", "")).strip()
    beats = script_plan.get("beats", [])
    context_reconstruction = _safe_dict(script_plan.get("context_reconstruction", {}))
    expected_product_type = str(script_plan.get("expected_product_type", "")).strip()
    grounded_product_type = str(script_plan.get("grounded_product_type", "")).strip()

    if expected_product_type and not grounded_product_type:
        issues.append("剧本缺少 grounded_product_type，无法确认创作内容是否围绕真实商品。")
    elif (
        expected_product_type
        and _canonical_product_type_for_review(expected_product_type)
        != _canonical_product_type_for_review(grounded_product_type)
    ):
        issues.append(
            "剧本 grounded_product_type 与商品身份卡不一致："
            f"expected={expected_product_type}, actual={grounded_product_type}。"
        )

    if not hook:
        issues.append("剧本缺少 hook。")
    if len(hook) > 80:
        issues.append("hook 过长，可能不适合短视频开头。")
    if not isinstance(body, list) or not [line for line in body if str(line).strip()]:
        issues.append("剧本缺少 body 卖点展开。")
    if not cta:
        issues.append("剧本缺少 CTA。")
    if not tone:
        issues.append("剧本缺少 tone 风格约束。")
    if len(rich_story_text) < 80:
        issues.append("剧本 rich_story_text 过短，缺少可供导演取舍的场景、动作和收束细节。")
    for field in ("scene_setting", "plot_development", "emotional_tendency", "speaking_intent"):
        if not str(context_reconstruction.get(field, "")).strip():
            issues.append(f"剧本缺少 context_reconstruction.{field}。")
    if not _string_list(context_reconstruction.get("causal_chain", [])):
        issues.append("剧本缺少 context_reconstruction.causal_chain。")
    if not isinstance(beats, list) or len(beats) < 3:
        issues.append("剧本 beats 过少，无法形成完整叙事闭环。")
    if isinstance(beats, list) and any(
        not str(beat.get("visual_intent", "")).strip()
        for beat in beats
        if isinstance(beat, dict)
    ):
        issues.append("剧本 beats 存在缺少 visual_intent 的节拍，导演无法据此设计画面。")
    if isinstance(beats, list):
        for index, beat in enumerate(beats, start=1):
            if not isinstance(beat, dict):
                issues.append(f"剧本 beat {index} 不是对象结构。")
                continue
            for field in ("scene_before", "action", "scene_after"):
                if not str(beat.get(field, "")).strip():
                    issues.append(f"剧本 beat {index} 缺少 {field}，无法形成可执行的状态变化。")
            for field in ("visible_entities", "physical_constraints", "asset_requirements"):
                if not _string_list(beat.get(field, [])):
                    issues.append(f"剧本 beat {index} 缺少 {field}，导演无法约束画面生成。")
            for field in (
                "shot_type",
                "camera_movement",
                "scene_description",
                "subject_appearance",
                "subject_position",
                "acting_direction",
                "dialogue",
                "cut_reason",
            ):
                if not str(beat.get(field, "")).strip():
                    issues.append(f"剧本 beat {index} 缺少 {field}。")
            if not _string_list(beat.get("scene_elements", [])):
                issues.append(f"剧本 beat {index} 缺少 scene_elements，无法追踪环境和道具状态。")

            continuity_mode = str(beat.get("continuity_mode", "")).strip()
            if continuity_mode not in {"new_scene", "continue_previous"}:
                issues.append(
                    f"剧本 beat {index} 的 continuity_mode 无效，必须是 new_scene 或 continue_previous。"
                )
            if continuity_mode == "new_scene" and not str(beat.get("transition_reason", "")).strip():
                issues.append(f"剧本 beat {index} 切换到新场景但缺少 transition_reason。")
            if index == 1 and continuity_mode == "continue_previous":
                issues.append("剧本第一个 beat 不能延续不存在的上一场景。")
            if index > 1 and continuity_mode == "continue_previous":
                previous_beat = beats[index - 2] if isinstance(beats[index - 2], dict) else {}
                previous_elements = set(_string_list(previous_beat.get("scene_elements", [])))
                current_elements = set(_string_list(beat.get("scene_elements", [])))
                if previous_elements and current_elements and not previous_elements.intersection(current_elements):
                    issues.append(
                        f"剧本 beat {index} 声明延续上一场景，但 scene_elements 与上一 beat 没有共同元素。"
                    )

    return _review_result(
        passed=not issues,
        issues=issues,
        retry_target="script_plan",
        retryable=True,
    )

def review_storyboard(
    storyboard: list[dict[str, Any]],
    product_context: dict[str, Any],
) -> dict[str, Any]:
    """审核分镜是否完整、可执行，并符合视频时长约束。"""

    issues: list[str] = []
    duration_limit = int(product_context.get("duration_seconds", 15))

    if not storyboard:
        issues.append("没有生成分镜。")
        return _review_result(False, issues, "storyboard", True)

    shot_count = len(storyboard)
    if shot_count < STORYBOARD_MIN_SHOTS or shot_count > STORYBOARD_MAX_SHOTS:
        issues.append(
            f"分镜数量为 {shot_count} 个，不在 {STORYBOARD_MIN_SHOTS}-{STORYBOARD_MAX_SHOTS} 个范围内，需要重新生成分镜。"
        )

    total_duration = 0
    crossfade_count = 0
    previous_shot: dict[str, Any] | None = None
    required_fields = [
        "shot_index",
        "duration_seconds",
        "purpose",
        "narrative_role",
        "scene_goal",
        "initial_state",
        "action",
        "final_state",
        "camera_motion",
        "visual_description",
        "subtitle",
        "voiceover",
        "asset_requirement",
        "render_strategy",
        "product_presence",
        "identity_strictness",
        "review_focus",
    ]
    for index, shot in enumerate(storyboard, start=1):
        for field_name in required_fields:
            if not str(shot.get(field_name, "")).strip():
                issues.append(f"第 {index} 个分镜缺少 {field_name}。")

        product_presence = str(shot.get("product_presence", "")).strip().lower()
        if product_presence not in ("required", "optional", "forbidden"):
            issues.append(f"第 {index} 个分镜 product_presence 值无效，应为 required/optional/forbidden。")
        identity_strictness = str(shot.get("identity_strictness", "")).strip().lower()
        if identity_strictness not in ("high", "medium", "low"):
            issues.append(f"第 {index} 个分镜 identity_strictness 值无效，应为 high/medium/low。")

        if product_presence == "required" and identity_strictness not in ("high", "medium"):
            issues.append(f"第 {index} 个分镜 product_presence=required 但 identity_strictness 不是 high 或 medium。")

        forbidden_variation = shot.get("forbidden_variation", [])
        if product_presence == "required" and not forbidden_variation:
            issues.append(f"第 {index} 个分镜 product_presence=required 但 forbidden_variation 为空。")

        review_focus = shot.get("review_focus", [])
        if product_presence == "required" and not review_focus:
            issues.append(f"第 {index} 个分镜 product_presence=required 但 review_focus 为空。")

        try:
            shot_duration = int(shot.get("duration_seconds", 0))
        except (TypeError, ValueError):
            shot_duration = 0

        if shot_duration <= 0:
            issues.append(f"第 {index} 个分镜时长无效。")
        total_duration += shot_duration

        transition_type = str(shot.get("transition_type", "hard_cut")).strip().lower()
        continuity_group = str(shot.get("continuity_group", "")).strip()
        previous_group = str((previous_shot or {}).get("continuity_group", "")).strip()
        if index > 1 and transition_type == "crossfade":
            crossfade_count += 1
        if transition_type == "continue_from_previous" and (
            previous_shot is None or not continuity_group or continuity_group != previous_group
        ):
            issues.append(
                f"第 {index} 个分镜使用 continue_from_previous，但没有与上一镜使用相同 continuity_group。"
            )
        if shot.get("anchor_last_frame") and transition_type != "continue_from_previous":
            issues.append(f"第 {index} 个分镜设置了 anchor_last_frame，但没有使用 continue_from_previous。")
        previous_shot = shot

    if total_duration > duration_limit:
        issues.append(f"分镜总时长 {total_duration} 秒超过限制 {duration_limit} 秒。")
    if crossfade_count > max(1, (len(storyboard) - 1) // 2):
        issues.append("crossfade 使用过多。跨场景默认应使用硬切，只有刻意表达氛围变化时才使用叠化。")

    narrative_roles = {
        str(shot.get("narrative_role", "")).strip().lower()
        for shot in storyboard
    }
    story_scene_roles = {"hook", "problem", "context", "lifestyle_result"}
    has_story_scene = any(
        role in story_scene_roles
        and str(shot.get("product_presence", "")).strip().lower() != "required"
        for role, shot in (
            (str(item.get("narrative_role", "")).strip().lower(), item)
            for item in storyboard
        )
    )
    if not has_story_scene:
        issues.append("分镜缺少独立剧情场景，不能全部是商品图片轻微运动。")
    if "cta" not in narrative_roles:
        issues.append("分镜缺少 CTA 收尾镜头。")

    return _review_result(
        passed=not issues,
        issues=issues,
        retry_target="storyboard",
        retryable=True,
    )



def _rule_based_narrative_review(
    product_context: dict[str, Any],
    script_plan: dict[str, Any],
    storyboard: list[dict[str, Any]],
) -> dict[str, Any]:
    """规则兜底的叙事闭环审核。"""

    issues: list[str] = []
    duration = int(product_context.get("duration_seconds", 15))
    contract = _safe_dict(script_plan.get("plan_contract", {}))
    strategy_family = _plan_strategy_family(script_plan, storyboard)

    full_subtitle = str(script_plan.get("full_subtitle_script", "")).strip()
    if len(full_subtitle) <= 20 and strategy_family == "legacy":
        issues.append("full_subtitle_script 缺失或过短，不像完整短视频文案。")

    beats = script_plan.get("beats", [])
    if beats:
        covered_start = min(int(b.get("start_seconds", 0)) for b in beats)
        covered_end = max(int(b.get("end_seconds", 0)) for b in beats)
        if covered_start > 0:
            issues.append("beats 未从 0 秒开始覆盖。")
        if covered_end < duration:
            issues.append(f"beats 仅覆盖到 {covered_end} 秒，未到总时长 {duration} 秒。")
    else:
        issues.append("缺少 beats 叙事节拍。")

    semantic_coverage = _string_list(script_plan.get("semantic_coverage", [])) or _semantic_coverage_from_storyboard(storyboard)
    required_coverage = _string_list(contract.get("required_coverage", []))
    if required_coverage:
        missing_coverage = [item for item in required_coverage if item not in semantic_coverage]
        if missing_coverage:
            issues.append("策略语义覆盖缺失：" + "、".join(missing_coverage))
    else:
        beat_roles = [str(b.get("role", "")).strip() for b in beats]
        if "hook" not in beat_roles:
            issues.append("缺少 hook 角色的 beat。")
        if "cta" not in beat_roles:
            issues.append("缺少 cta 角色的 beat。")

    # 检查分镜间的承接关系，但不作为阻断条件。
    # LLM 产出的 initial_state / final_state 经常为空，但内容本身仍然可用。
    # 这里仅记录提示，不阻止流程继续。
    continuity_gaps = []
    if storyboard and len(storyboard) > 1:
        for i in range(len(storyboard) - 1):
            final_state = str(storyboard[i].get("final_state", "")).strip()
            next_initial = str(storyboard[i + 1].get("initial_state", "")).strip()
            if not final_state and not next_initial:
                continuity_gaps.append(f"第 {i + 1} 和第 {i + 2} 个分镜之间缺少承接关系。")
                break

    return {
        "passed": not issues,
        "issues": issues,
        "suggestions": continuity_gaps,
        "strategy_family": strategy_family,
        "semantic_coverage": semantic_coverage,
        "plan_contract": contract,
    }


def _allocate_assets_to_shot_roles(
    asset_profiles: list[dict[str, Any]],
    assets: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """根据asset_profiles的suitable_for和visual_role，把不同素材分配到不同镜头职责。

    Returns:
        {
            "appearance_anchor": [asset_id1, asset_id2, ...],  # 整机外观→product_reveal/product_hero/cta
            "detail_reference": [asset_id3, ...],              # 细节→feature_demo/detail_closeup
            "scene_context": [asset_id4, ...],                 # 使用场景→usage/lifestyle
        }
    """

    allocation: dict[str, list[str]] = {
        "appearance_anchor": [],
        "detail_reference": [],
        "scene_context": [],
    }

    assets_by_id = {str(a.get("asset_id", "")): a for a in assets}

    for profile in asset_profiles:
        asset_id = str(profile.get("asset_id", "")).strip()
        if not asset_id or asset_id not in assets_by_id:
            continue

        visual_role = str(profile.get("visual_role", "")).strip()
        suitable_for = profile.get("suitable_for", [])

        # 按visual_role优先分配
        roles = _string_list(profile.get("normalized_roles", []))
        capabilities = _safe_dict(profile.get("material_capabilities"))

        if (
            visual_role in ("appearance_anchor", "full_product_anchor")
            or "appearance_anchor_candidate" in roles
            or capabilities.get("appearance_anchor_candidate")
        ):
            allocation["appearance_anchor"].append(asset_id)
        elif visual_role in ("detail_reference", "logo_detail", "brand_detail"):
            allocation["detail_reference"].append(asset_id)
        elif visual_role == "scene_context":
            allocation["scene_context"].append(asset_id)
        # 如果visual_role未明确，按suitable_for推断
        elif "detail_closeup" in suitable_for or "feature_detail" in suitable_for:
            allocation["detail_reference"].append(asset_id)
        elif "usage_scene" in suitable_for or "lifestyle" in suitable_for or "scene_context" in suitable_for:
            allocation["scene_context"].append(asset_id)
        else:
            # 默认作为外观锚点候选
            allocation["appearance_anchor"].append(asset_id)

    return allocation


def match_assets_to_storyboard(
    storyboard: list[dict[str, Any]],
    asset_analysis: dict[str, Any],
) -> list[dict[str, Any]]:
    """把可用素材匹配到每个分镜。"""

    print("[video_generation_workflow] 开始匹配分镜素材。", flush=True)
    assets = asset_analysis.get("assets", [])
    image_assets = [asset for asset in assets if asset.get("asset_type") == "image" and asset.get("is_supported")]
    asset_profiles = asset_analysis.get("asset_profiles", [])
    shared_scene_asset = _shared_scene_background_asset(asset_analysis)
    force_opening_anchor = len(storyboard) > 1
    first_frame_anchor = _best_first_frame_anchor(image_assets) if force_opening_anchor else None
    first_shot_position = _first_required_product_storyboard_position(storyboard) if force_opening_anchor else -1

    # 阶段A：按素材理解分配职责，避免一张图复用到所有商品镜
    asset_allocation = _allocate_assets_to_shot_roles(asset_profiles, image_assets)
    used_asset_ids: set[str] = set()  # 追踪已用过的素材，避免单张复用
    matches: list[dict[str, Any]] = []

    for index, shot in enumerate(storyboard):
        product_presence = str(shot.get("product_presence", "")).strip().lower()
        matched_asset = None
        is_first_frame_shot = index == first_shot_position
        if is_first_frame_shot and first_frame_anchor:
            matched_asset = first_frame_anchor
            shot["asset_id"] = first_frame_anchor.get("asset_id", "")
            shot["render_strategy"] = "image_to_video"
            shot["product_presence"] = "required"
            shot["identity_strictness"] = "high"
            shot["narrative_role"] = shot.get("narrative_role") or "product_reveal"
            if not _should_preserve_first_product_shot_contract(shot):
                shot["initial_state"] = "第一帧必须是用户上传素材图中的真实商品主体，不从人物剧情或文生场景开场。"
                shot["scene_goal"] = "第一秒先建立真实商品身份，再展开后续剧情或卖点。"
        # 导演已经为商品镜指定真实素材时，必须优先使用该素材。
        # 统一棚拍背景只能作为没有明确商品锚点时的兜底，不能截胡真实商品图。
        if product_presence != "forbidden" and not matched_asset:
            matched_asset = _resolve_storyboard_asset(shot, image_assets)
        # 导演可能为整机镜头误选 Logo 或键盘局部图。存在完整商品素材时，
        # 渲染入口必须切换到完整外观锚点；局部素材只留给细节镜头。
        if matched_asset and _shot_requires_full_product_anchor(shot) and not _is_full_product_anchor(matched_asset):
            matched_asset = _best_full_product_asset(image_assets) or matched_asset
        # CTA 是最终收束画面。只要存在完整商品图，就必须使用真实整机锚点，
        # 不能让纯文生视频在最后几秒重新发明一个外观或 Logo。
        if (
            product_presence != "forbidden"
            and not matched_asset
            and str(shot.get("narrative_role", "")).strip().lower() == "cta"
        ):
            matched_asset = _best_full_product_asset(image_assets)
        if product_presence != "forbidden" and not matched_asset and _shot_can_use_shared_scene_background(shot):
            matched_asset = shared_scene_asset
        # 阶段A新逻辑：根据镜头角色和素材职责分配，优先用未使用过的素材
        if product_presence != "forbidden" and not matched_asset and _shot_prefers_real_asset(shot) and image_assets:
            narrative_role = str(shot.get("narrative_role", "")).strip().lower()
            candidate_pool: list[dict[str, Any]] = []

            # 按镜头角色选择候选池
            if narrative_role in ("product_reveal", "product_hero", "cta"):
                # 整机外观镜：优先用appearance_anchor池中未用过的
                for asset_id in asset_allocation.get("appearance_anchor", []):
                    if asset_id not in used_asset_ids:
                        for asset in image_assets:
                            if str(asset.get("asset_id", "")) == asset_id:
                                candidate_pool.append(asset)
                                break
            elif narrative_role in ("feature_demo", "detail_proof", "detail_closeup"):
                # 细节镜：优先用detail_reference池
                for asset_id in asset_allocation.get("detail_reference", []):
                    if asset_id not in used_asset_ids:
                        for asset in image_assets:
                            if str(asset.get("asset_id", "")) == asset_id:
                                candidate_pool.append(asset)
                                break
                # 细节池空了，再从外观池取
                if not candidate_pool:
                    for asset_id in asset_allocation.get("appearance_anchor", []):
                        if asset_id not in used_asset_ids:
                            for asset in image_assets:
                                if str(asset.get("asset_id", "")) == asset_id:
                                    candidate_pool.append(asset)
                                    break
            elif narrative_role in ("usage", "lifestyle", "usage_or_lifestyle"):
                # 使用场景镜：优先scene_context池
                for asset_id in asset_allocation.get("scene_context", []):
                    if asset_id not in used_asset_ids:
                        for asset in image_assets:
                            if str(asset.get("asset_id", "")) == asset_id:
                                candidate_pool.append(asset)
                                break

            # 候选池有未用素材，选最佳；否则回退旧逻辑
            if candidate_pool:
                matched_asset = _best_asset_with_geometric_fallback(candidate_pool)
            else:
                # 所有分配池都用完或无匹配，回退按角色选
                matched_asset = _best_asset_for_shot(shot, image_assets) or image_assets[index % len(image_assets)]
        strategy = _choose_render_strategy(shot, matched_asset)
        render_asset = _select_render_asset_variant(shot, matched_asset)
        if matched_asset and _should_use_full_frame_asset(shot):
            render_asset = _full_frame_render_asset(matched_asset)
        render_input = _build_render_input(render_asset, strategy)
        reference_scope = _asset_reference_scope(render_asset)
        creative_completion_required = bool(
            render_asset
            and not render_asset.get("is_scene_background")
            and _shot_requires_full_product_anchor(shot)
            and not _is_full_product_anchor(render_asset)
        )
        # 记录已使用的素材，避免下一个商品镜重复使用同一张
        if matched_asset and matched_asset.get("asset_id"):
            used_asset_ids.add(str(matched_asset.get("asset_id", "")))

        matches.append(
            {
                "shot_index": shot["shot_index"],
                "strategy": strategy,
                "matched_asset": render_asset,
                "source_asset": matched_asset,
                "render_input": render_input,
                "reference_scope": reference_scope,
                "creative_completion_required": creative_completion_required,
                "match_status": "matched" if render_asset else "generated_without_asset",
                "note": _asset_match_note(strategy, render_asset),
            }
        )

    print("[video_generation_workflow] 分镜素材匹配完成。", flush=True)
    return matches


def _shared_scene_background_asset(asset_analysis: dict[str, Any]) -> dict[str, Any] | None:
    """把预处理生成的空棚拍底图包装为渲染层可消费的场景素材。"""

    file_path = str(asset_analysis.get("shared_scene_background_path", "")).strip()
    if not file_path:
        return None
    reveal_asset = _best_appearance_anchor_asset(asset_analysis.get("assets", []))
    reveal_asset_path = str((reveal_asset or {}).get("anchor_file_path", "")).strip()
    return {
        "asset_id": "shared_scene_background",
        "asset_type": "image",
        "file_path": file_path,
        "filename": "统一棚拍背景",
        "is_supported": True,
        "is_scene_background": True,
        "reveal_asset_path": reveal_asset_path,
    }


def _shot_can_use_shared_scene_background(shot: dict[str, Any]) -> bool:
    """只有商品揭示桥接镜使用共享场景底图，剧情镜保留独立画面。"""

    narrative_role = str(shot.get("narrative_role", "")).strip().lower()
    continuity_mode = str(shot.get("continuity_mode", "")).strip().lower()
    return narrative_role == "product_reveal" or continuity_mode == "shared_scene_bridge"


def _resolve_storyboard_asset(shot: dict[str, Any], image_assets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """按导演分镜指定的 asset_id 精确匹配上传素材。"""

    requested_ids = []
    direct_asset_id = str(shot.get("asset_id", "")).strip()
    if direct_asset_id:
        requested_ids.append(direct_asset_id)

    asset_usage = shot.get("asset_usage") or {}
    for asset_id in asset_usage.get("selected_asset_ids", []):
        asset_id = str(asset_id).strip()
        if asset_id and asset_id not in requested_ids:
            requested_ids.append(asset_id)

    for requested_id in requested_ids:
        for asset in image_assets:
            if str(asset.get("asset_id", "")).strip() == requested_id:
                return asset
    return None


def _best_asset_for_shot(shot: dict[str, Any], image_assets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """没有精确绑定时按镜头角色选素材，避免把局部特写当成整机外观锚点。"""

    if not image_assets:
        return None
    narrative_role = str(shot.get("narrative_role", "")).strip().lower()
    if narrative_role == "detail_proof":
        preferred_roles = {"detail_reference", "appearance_anchor"}
    else:
        preferred_roles = {"appearance_anchor"}
    preferred = [
        asset for asset in image_assets if str(asset.get("visual_role", "")).strip() in preferred_roles
    ]
    return _best_asset_with_geometric_fallback(preferred or image_assets)


def _best_full_product_asset(image_assets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """从可用图片中选择质量最高的完整商品素材。"""

    candidates = [asset for asset in image_assets if _is_full_product_anchor(asset)]
    if not candidates:
        return None
    return _best_asset_with_geometric_fallback(candidates)


def _best_first_frame_anchor(image_assets: list[dict[str, Any]]) -> dict[str, Any] | None:
    """选择首帧强锚点：优先完整商品图，退而使用最佳外观锚点。"""

    if not image_assets:
        return None
    return (
        _best_full_product_asset(image_assets)
        or _best_appearance_anchor_asset(image_assets)
        or _best_asset_with_geometric_fallback(image_assets)
    )


def _first_storyboard_position(storyboard: list[dict[str, Any]]) -> int:
    """返回时间排序后的第一镜位置，兼容 0/1 起始编号。"""

    if not storyboard:
        return -1
    return min(
        range(len(storyboard)),
        key=lambda index: int(storyboard[index].get("shot_index", index) or index),
    )


def _first_required_product_storyboard_position(storyboard: list[dict[str, Any]]) -> int:
    """返回第一个明确展示真实商品的镜头位置。"""

    required_positions = [
        index
        for index, shot in enumerate(storyboard)
        if str(shot.get("product_presence", "")).strip().lower() == "required"
    ]
    if not required_positions:
        return _first_storyboard_position(storyboard)
    return min(
        required_positions,
        key=lambda index: int(storyboard[index].get("shot_index", index) or index),
    )


def _asset_reference_scope(asset: dict[str, Any] | None) -> str:
    """记录渲染锚点的约束范围，避免把局部图误解为整机复刻依据。"""

    if not asset:
        return "none"
    if asset.get("is_scene_background"):
        return "scene"
    if _is_full_product_anchor(asset):
        return "full_product"
    if str(asset.get("visual_role", "")).strip() == "detail_reference":
        return "detail"
    return "unknown"



def _select_render_asset_variant(shot: dict[str, Any], matched_asset: dict[str, Any] | None) -> dict[str, Any] | None:
    """根据分镜角色选择预处理后的弱关键帧，减少同一素材反复拉伸的 PPT 感。"""

    if not matched_asset:
        return None
    if _should_use_full_frame_asset(shot) and _has_explicit_full_frame_asset(matched_asset):
        return _full_frame_render_asset(matched_asset)
    variants = matched_asset.get("keyframe_variants") or {}
    if not isinstance(variants, dict) or not variants:
        return matched_asset

    role = str(shot.get("narrative_role", "")).strip().lower()
    review_text = " ".join(str(i) for i in shot.get("review_focus", []) or [])
    visual_text = " ".join([
        role,
        str(shot.get("scene_goal", "")),
        str(shot.get("visual_description", "")),
        review_text,
    ]).lower()

    variant_key = "hero"
    material_strategy = str(shot.get("material_strategy", "")).strip()
    selected_skill = str(shot.get("selected_prompt_skill", "")).strip()
    if material_strategy == "detail_reference" or selected_skill.startswith("detail_reference"):
        variant_key = "detail"
    # 商品揭示镜必须先给出完整主体，不能因为文案顺带提到标识就误选成局部特写。
    elif role in {"product_reveal", "product_hero", "feature_demo"}:
        variant_key = "hero"
    elif role == "cta":
        # CTA 文案由本地后处理叠加。缩小商品会损失结构和商标细节，
        # 因此视频模型仍使用完整 hero 锚点。
        variant_key = "hero"
    elif role == "detail_proof" or any(k in visual_text for k in ("logo", "商标", "标识", "材质", "细节", "接口", "纹理")):
        variant_key = "detail"

    variant_path = str(variants.get(variant_key, "")).strip()
    if not variant_path or not Path(variant_path).exists():
        return matched_asset

    render_asset = dict(matched_asset)
    render_asset["variant_source_file_path"] = matched_asset.get("file_path", "")
    render_asset["file_path"] = variant_path
    render_asset["keyframe_variant"] = variant_key
    return render_asset


def _should_use_full_frame_asset(shot: dict[str, Any]) -> bool:
    """完整商品镜头必须使用完整上传图，避免抠图把 logo 当成商品主体。"""

    if bool(shot.get("force_full_frame_anchor")):
        return True
    if _shot_requires_full_product_anchor(shot):
        return True
    product_presence = str(shot.get("product_presence", "")).strip().lower()
    identity_strictness = str(shot.get("identity_strictness", "")).strip().lower()
    return product_presence == "required" and identity_strictness == "high"


def _has_explicit_full_frame_asset(matched_asset: dict[str, Any]) -> bool:
    """判断素材是否保留了标准化/原始完整图路径。"""

    return bool(
        str(matched_asset.get("standardized_file_path", "")).strip()
        or str(matched_asset.get("original_file_path", "")).strip()
    )


def _full_frame_render_asset(matched_asset: dict[str, Any]) -> dict[str, Any]:
    """返回使用标准化完整图的渲染素材，不使用抠图/局部 keyframe。"""

    full_frame_path = (
        str(matched_asset.get("original_file_path", "")).strip()
        or str(matched_asset.get("standardized_file_path", "")).strip()
        or str(matched_asset.get("file_path", "")).strip()
    )
    if not full_frame_path or not Path(full_frame_path).exists():
        return matched_asset
    render_asset = dict(matched_asset)
    render_asset["cropped_anchor_file_path"] = matched_asset.get("file_path", "")
    render_asset["file_path"] = full_frame_path
    render_asset["keyframe_variant"] = "full_frame"
    render_asset["is_full_frame_anchor"] = True
    return render_asset

def _build_render_input(matched_asset: dict[str, Any] | None, strategy: str) -> dict[str, Any] | None:
    """把素材匹配结果转换成渲染层可执行的输入合同。"""

    if not matched_asset or not matched_asset.get("file_path"):
        return None
    if strategy not in {"image_to_video", "crop_and_ken_burns"}:
        return None
    return {
        "type": "asset",
        "asset_id": matched_asset.get("asset_id", ""),
        "file_path": matched_asset.get("file_path", ""),
        "asset_type": matched_asset.get("asset_type", ""),
        "is_scene_background": bool(matched_asset.get("is_scene_background")),
        "reveal_asset_path": matched_asset.get("reveal_asset_path", ""),
        "render_mode": strategy,
        "fallback_policy": "local_asset_motion",
    }


def _choose_gap_strategy(shot: dict[str, Any], matched_asset: dict[str, Any] | None, product_identity_card: dict[str, Any]) -> str:
    product_presence = shot.get("product_presence", "optional")

    if matched_asset and matched_asset.get("file_path"):
        return "use_existing_asset"

    if product_presence == "required":
        if product_identity_card.get("appearance_anchor_available"):
            return "use_appearance_anchor_image_to_video"
        else:
            return "use_uploaded_asset_ken_burns"

    if product_presence == "optional" or product_presence == "forbidden":
        return "text_to_video_generic_scene"

    return "text_to_video_generic_scene"


def _ken_burns_fallback(shot: dict[str, Any], asset: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "strategy": "ken_burns",
        "reason": "视频模型多次生成失败或商品漂移，降级为上传素材图 + Ken Burns 运镜",
        "asset": asset,
        "camera_motion": shot.get("camera_motion", "推近"),
    }


def complete_asset_gaps(
    storyboard: list[dict[str, Any]],
    asset_matching: list[dict[str, Any]],
    asset_analysis: dict[str, Any],
    product_identity_card: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把分镜里的素材缺口转换成可执行方案，并保留每个缺口的处理记录。"""

    if product_identity_card is None:
        product_identity_card = asset_analysis.get("product_identity_card", {})
    product_identity_card = _safe_dict(product_identity_card)
    assets = asset_analysis.get("assets", [])
    product_assets = [
        asset
        for asset in assets
        if asset.get("asset_type") == "image" and asset.get("is_supported") and asset.get("file_path")
    ]
    shots_by_index = {shot["shot_index"]: shot for shot in storyboard}
    completed_matches: list[dict[str, Any]] = []
    gap_records: list[dict[str, Any]] = []

    for match in asset_matching:
        updated_match = dict(match)
        shot = shots_by_index.get(match.get("shot_index"), {})
        strategy = str(match.get("strategy", ""))
        matched_asset = match.get("matched_asset")
        gap_strategy = _choose_gap_strategy(shot, matched_asset, product_identity_card)
        updated_match["gap_strategy"] = gap_strategy

        # ai_image_then_video 表示"最好先补一张场景图"。当前没有文生图 endpoint，
        # 所以先把它降级成文生视频，并保留 prompt，后续接文生图时不用改上游分镜结构。
        if strategy == "ai_image_then_video":
            updated_match.update(
                {
                    "strategy": "text_to_video",
                    "match_status": "gap_completed",
                    "completion_type": "text_to_video_prompt",
                    "generated_prompt": _build_gap_completion_prompt(shot),
                    "note": "缺少对应场景图，当前用分镜画面描述走文生视频；后续可替换为文生图后再图生视频。",
                }
            )
            gap_records.append(_asset_gap_record(shot, strategy, updated_match))

        elif strategy == "needs_user_asset":
            if product_assets:
                anchor_asset = _resolve_storyboard_asset(shot, product_assets) or product_assets[0]
                if gap_strategy == "use_appearance_anchor_image_to_video":
                    updated_match.update(
                        {
                            "strategy": "image_to_video",
                            "matched_asset": _select_render_asset_variant(shot, anchor_asset),
                            "render_input": _build_render_input(_select_render_asset_variant(shot, anchor_asset), "image_to_video"),
                            "match_status": "gap_completed",
                            "completion_type": "appearance_anchor_binding",
                            "note": f"商品主镜头，已绑定 {anchor_asset.get('filename', '商品图')} 作为 appearance_anchor。",
                        }
                    )
                elif gap_strategy == "use_uploaded_asset_ken_burns":
                    ken_burns = _ken_burns_fallback(shot, anchor_asset)
                    updated_match.update(
                        {
                            "strategy": "crop_and_ken_burns",
                            "matched_asset": _select_render_asset_variant(shot, anchor_asset),
                            "render_input": _build_render_input(_select_render_asset_variant(shot, anchor_asset), "crop_and_ken_burns"),
                            "match_status": "gap_completed_with_risk",
                            "completion_type": "ken_burns_fallback",
                            "risk": "缺少 appearance_anchor，商品主镜头降级为 Ken Burns 运镜。",
                            "note": ken_burns["reason"],
                            "camera_motion": ken_burns["camera_motion"],
                        }
                    )
                else:
                    updated_match.update(
                        {
                            "strategy": "text_to_video",
                            "match_status": "gap_completed",
                            "completion_type": "scene_text_to_video",
                            "generated_prompt": _build_gap_completion_prompt(shot),
                            "note": "场景铺垫镜头，允许文生视频生成通用场景。",
                        }
                    )
            else:
                if gap_strategy == "use_uploaded_asset_ken_burns" or gap_strategy == "use_appearance_anchor_image_to_video":
                    ken_burns = _ken_burns_fallback(shot, None)
                    updated_match.update(
                        {
                            "strategy": "crop_and_ken_burns",
                            "match_status": "gap_completed_with_risk",
                            "completion_type": "ken_burns_fallback",
                            "risk": "缺少真实商品图，商品主镜头降级为 Ken Burns 运镜，保证商品一致性。",
                            "note": "商品主镜头不能凭空生成另一个商品，已降级为 Ken Burns。",
                            "camera_motion": ken_burns["camera_motion"],
                        }
                    )
                else:
                    updated_match.update(
                        {
                            "strategy": "text_to_video",
                            "match_status": "gap_completed_with_risk",
                            "completion_type": "model_generated_missing_asset",
                            "generated_prompt": _build_gap_completion_prompt(shot),
                            "risk": "缺少真实商品图，视频模型会根据商品信息生成缺失画面，可能和真实外观存在偏差。",
                            "note": "非商品主镜头，允许文生视频生成通用场景。",
                        }
                    )
            gap_records.append(_asset_gap_record(shot, strategy, updated_match))

        elif strategy == "text_to_video" and not matched_asset:
            if str(shot.get("product_presence", "optional")).strip().lower() == "required":
                updated_match.update(
                    {
                        "strategy": "needs_user_asset",
                        "match_status": "unresolved",
                        "completion_type": "blocked_required_product_asset",
                        "risk": "商品主镜头缺少真实商品素材，已阻断文生视频生成，避免 logo 或外观错误。",
                        "note": "请补充商品主图、侧面图或使用场景图；也可以手动降级为无商品场景。",
                    }
                )
                gap_records.append(_asset_gap_record(shot, strategy, updated_match))
            else:
                updated_match.update(
                    {
                        "match_status": "generated_without_asset",
                        "completion_type": "text_to_video_prompt",
                        "generated_prompt": _build_gap_completion_prompt(shot),
                    }
                )

        if updated_match.get("matched_asset") and not updated_match.get("render_input"):
            updated_match["render_input"] = _build_render_input(
                updated_match.get("matched_asset"),
                str(updated_match.get("strategy", "")),
            )

        completed_matches.append(updated_match)

    risk_count = sum(1 for item in completed_matches if item.get("match_status") == "gap_completed_with_risk")
    unresolved_count = sum(1 for item in completed_matches if item.get("match_status") == "unresolved")
    return {
        "asset_matching": completed_matches,
        "gap_records": gap_records,
        "completed_count": len(gap_records),
        "unresolved_count": unresolved_count,
        "risk_count": risk_count,
        "note": "商品主镜头缺少真实素材时不再交给文生视频硬生成；系统会绑定可用外观锚点、降级 Ken Burns，或标记为 unresolved 等待补素材。",
    }


def _build_video_prompt_constraints(
    product_identity_card: dict[str, Any],
    conservative_constraints: dict[str, Any],
) -> dict[str, list[str]]:
    """构造视频 Prompt 约束，保持必须继承项和禁止项语义分离。"""

    must_preserve = list(product_identity_card.get("must_preserve", []))
    must_avoid = list(product_identity_card.get("forbidden_changes", []))
    if conservative_constraints:
        must_preserve.append("优先使用上传素材作为参考")
        must_avoid.append("复杂动作")
    return {
        "must_preserve": must_preserve,
        "must_avoid": must_avoid,
    }


def _first_frame_anchor_from_asset_matching(asset_matching: list[dict[str, Any]]) -> dict[str, Any] | None:
    """从素材匹配结果中选择最终创作计划的首帧真实素材锚点。"""

    image_assets: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for match in asset_matching:
        for key in ("source_asset", "matched_asset"):
            asset = match.get(key)
            if not isinstance(asset, dict) or asset.get("is_scene_background"):
                continue
            file_path = str(asset.get("file_path", "")).strip()
            if not file_path or file_path in seen_paths:
                continue
            seen_paths.add(file_path)
            image_assets.append(asset)
    return _best_first_frame_anchor(image_assets)


def build_creation_plan(
    product_context: dict[str, Any],
    storyboard: list[dict[str, Any]],
    asset_matching: list[dict[str, Any]],
) -> dict[str, Any]:
    """生成后续视频创作模块可以消费的执行计划。"""

    print("[video_generation_workflow] 开始生成创作计划。", flush=True)
    shots = []
    matches_by_index = {item["shot_index"]: item for item in asset_matching}
    force_opening_anchor = len(storyboard) > 1
    first_frame_anchor = _first_frame_anchor_from_asset_matching(asset_matching) if force_opening_anchor else None
    first_shot_position = _first_required_product_storyboard_position(storyboard) if force_opening_anchor else -1
    product_identity_card = product_context.get("product_identity_card", {})
    visual_style_bible = product_context.get("visual_style_bible") or {
        "realism": "真实写实的商业短视频",
        "lighting": "柔和自然光，主体照明稳定",
        "color_temperature": "中性偏暖色温",
        "background_complexity": "背景克制干净",
        "camera_language": "稳定镜头，运动幅度小",
    }
    render_segments = adapt_storyboard_to_render_segments(storyboard)
    render_segments_by_index = {segment["shot_index"]: segment for segment in render_segments}

    for shot_position, shot in enumerate(storyboard):
        match = matches_by_index.get(shot["shot_index"], {})
        render_segment = render_segments_by_index.get(shot["shot_index"], {})
        shot_purpose = _shot_purpose(shot)
        matched_asset = match.get("matched_asset")
        is_first_shot = shot_position == first_shot_position
        preserve_first_contract = bool(
            is_first_shot
            and first_frame_anchor
            and _should_preserve_first_product_shot_contract(shot)
        )
        if is_first_shot and first_frame_anchor:
            matched_asset = _full_frame_render_asset(first_frame_anchor)
            match = dict(match)
            match["matched_asset"] = matched_asset
            match["source_asset"] = first_frame_anchor
            match["strategy"] = "image_to_video"
            match["render_input"] = _build_render_input(matched_asset, "image_to_video")
            match["reference_scope"] = _asset_reference_scope(matched_asset)
        render_strategy = str(match.get("strategy", "text_to_video"))
        product_presence_override = shot.get("product_presence", "optional")
        if is_first_shot and first_frame_anchor:
            render_strategy = "image_to_video"
            product_presence_override = "required"
        blocked_missing_product_asset = render_strategy == "needs_user_asset" and str(shot.get("product_presence", "optional")).strip().lower() == "required"
        if blocked_missing_product_asset:
            # 没有商品真实锚点时，不让视频模型凭空生成商品；退化为无商品铺垫镜并在 gap 里提示补素材。
            render_strategy = "text_to_video"
            product_presence_override = "forbidden"
        if (
            matched_asset
            and matched_asset.get("file_path")
            and render_strategy == "text_to_video"
            and str(product_presence_override).strip().lower() != "forbidden"
        ):
            render_strategy = "image_to_video"
            product_presence_override = shot.get("product_presence", product_presence_override)
        selected_prompt_skill = str(shot.get("selected_prompt_skill", "")).strip()
        allows_text_reconstruction = selected_prompt_skill == "commerce_scene.new_scene_result"
        if (
            render_strategy == "text_to_video"
            and not matched_asset
            and product_presence_override != "required"
            and not allows_text_reconstruction
        ):
            # 无真实锚点的文生视频只负责铺垫场景，禁止出现可识别商品，避免模型凭空发明品牌和 Logo。
            product_presence_override = "forbidden"
        preserve_identity_tail = bool(
            render_strategy == "image_to_video"
            and match.get("reference_scope") == "full_product"
            and _shot_requires_full_product_anchor(shot)
        )
        render_input = match.get("render_input")
        # 真实素材存在时，执行合同必须与最终策略一致。不能沿用上游残留的纯文本输入。
        if matched_asset and matched_asset.get("file_path"):
            if not isinstance(render_input, dict) or render_input.get("type") != "asset":
                render_input = _build_render_input(matched_asset, render_strategy)
        if not render_input:
            render_input = _build_render_input(matched_asset, render_strategy)
        force_video_prompt = bool(shot.get("force_video_prompt") and str(shot.get("video_prompt", "")).strip())
        video_prompt_constraints = (
            {}
            if force_video_prompt
            else _build_video_prompt_constraints(product_identity_card, product_context.get("conservative_constraints", {}))
        )
        downstream_identity_constraints = [] if force_video_prompt else shot.get(
            "product_identity_constraints",
            product_identity_card.get("must_preserve", []),
        )
        downstream_conservative_constraints = {} if force_video_prompt else product_context.get("conservative_constraints", {})
        downstream_allowed_variation = [] if force_video_prompt else shot.get("allowed_variation", [])
        downstream_forbidden_variation = [] if force_video_prompt else shot.get("forbidden_variation", [])
        downstream_identity_card = {} if force_video_prompt else product_identity_card
        shots.append(
            {
                "shot_index": shot["shot_index"],
                "duration_seconds": shot["duration_seconds"],
                "purpose": shot_purpose,
                "narrative_role": shot.get("narrative_role", ""),
                "continuity_mode": shot.get("continuity_mode", ""),
                "continuity_group": shot.get("continuity_group", ""),
                "transition_type": shot.get("transition_type", "hard_cut"),
                "anchor_last_frame": bool(shot.get("anchor_last_frame")),
                "force_full_frame_anchor": bool(is_first_shot and first_frame_anchor),
                # 对整机展示镜使用同一张真实商品图约束首尾帧，减少 Logo 和机身在中间过程漂移。
                "preserve_identity_tail": preserve_identity_tail,
                "visual_style_bible": visual_style_bible,
                "scene_goal": (
                    "第一秒先建立真实商品身份：画面从上传素材图中的同一件商品开始，再轻微展开卖点。"
                    if is_first_shot and first_frame_anchor and not preserve_first_contract
                    else shot.get("scene_goal", shot_purpose)
                ),
                "initial_state": (
                    "第一帧必须严格等于上传素材图的商品画面；商品主体、颜色、结构、logo 和数量都以素材为准。"
                    if is_first_shot and first_frame_anchor and not preserve_first_contract
                    else shot.get("initial_state", "")
                ),
                "action": (
                    "只允许轻微镜头推进、自然光影变化或商品本体极轻微稳定动作；不要用人物剧情开场。"
                    if is_first_shot and first_frame_anchor and not preserve_first_contract
                    else shot.get("action", "")
                ),
                "final_state": shot.get("final_state", ""),
                "camera_motion": shot.get("camera_motion", ""),
                "visual_description": (
                    "从上传素材图的真实商品首帧开始，商品占据画面主体，保持素材中的同一件商品，背景和光影可轻微商业化优化；禁止先出现人物、无关剧情或重新生成类似商品。"
                    if is_first_shot and first_frame_anchor and not preserve_first_contract
                    else shot.get("visual_description", shot.get("seedance_prompt", "")[:100])
                ),
                "subtitle": shot.get("subtitle", shot.get("voiceover", "")),
                "voiceover": shot.get("voiceover", shot.get("subtitle", "")),
                "product_identity_constraints": downstream_identity_constraints,
                "conservative_constraints": downstream_conservative_constraints,
                "asset_usage": shot.get("asset_usage", {}),
                "material_strategy": shot.get("material_strategy", ""),
                "selected_prompt_skill": shot.get("selected_prompt_skill", ""),
                "required_for_variant": bool(shot.get("required_for_variant")),
                "asset_usage_reason": shot.get("asset_usage_reason", ""),
                "planner_source": shot.get("planner_source", ""),
                "generation_mode": shot.get("generation_mode", match.get("strategy", "text_to_video")),
                "force_video_prompt": force_video_prompt,
                "video_prompt": shot.get("video_prompt", "") if force_video_prompt else "",
                "final_prompt_source": shot.get("final_prompt_source", ""),
                "video_prompt_constraints": video_prompt_constraints,
                "risk_notes": shot.get("risk_notes", []),
                "product_presence": product_presence_override,
                "identity_strictness": "high" if is_first_shot and first_frame_anchor else shot.get("identity_strictness", "low"),
                "allowed_variation": downstream_allowed_variation,
                "forbidden_variation": downstream_forbidden_variation,
                "review_focus": shot.get("review_focus", []),
                "completion_criteria": shot.get("completion_criteria", []),
                "product_identity_card": downstream_identity_card,
                "render_segment": render_segment,
                "asset_id": shot.get("asset_id", "") or (matched_asset or {}).get("asset_id", ""),
                "asset": matched_asset,
                "render_input": render_input or {
                    "type": "text",
                    "prompt": (
                        "真实生活场景，只出现普通无品牌道具，不出现可识别商品主体、logo 或品牌文字；通过字幕和口播表达卖点。"
                        if blocked_missing_product_asset
                        else match.get("generated_prompt") or shot.get("visual_description", "")
                    ),
                },
                "render_strategy": render_strategy,
                "render_reason": (
                    "首个商品镜使用上传素材图作为第一帧锚点，并保留上游 material-aware 动作契约。"
                    if preserve_first_contract
                    else "首镜强制使用上传素材图作为第一帧锚点。"
                    if is_first_shot and first_frame_anchor
                    else match.get("note", "")
                ),
                # 模板路径B产出的直接seedance_prompt，透传给渲染器，不再二次处理
                "seedance_prompt": shot.get("seedance_prompt", ""),
                "asset_file": shot.get("asset_file", ""),
            }
        )

    plan = {
        "target_platform": product_context.get("target_platform", "tiktok"),
        "aspect_ratio": "9:16",
        "total_duration_seconds": sum(segment["target_duration_seconds"] for segment in render_segments),
        "render_mode": "seedance_auto_with_local_fallback",
        "visual_style_bible": visual_style_bible,
        "render_segments": render_segments,
        "shots": shots,
        "next_module": "seedance_video_rendering",
    }
    print("[video_generation_workflow] 创作计划生成完成。", flush=True)
    return plan


def _shot_has_explicit_material_plan(shot: dict[str, Any]) -> bool:
    if str(shot.get("material_strategy", "")).strip():
        return True
    if str(shot.get("selected_prompt_skill", "")).strip():
        return True
    planner_source = str(shot.get("planner_source", "")).strip()
    return "source_scene_extension" in planner_source or "material_first" in planner_source


def _shot_has_detailed_motion_contract(shot: dict[str, Any]) -> bool:
    action = str(shot.get("action", "")).strip()
    visual = str(shot.get("visual_description", "")).strip()
    if len(action) < 18 or len(visual) < 40:
        return False
    generic_fragments = ("自然光影变化", "轻微推进", "保持稳定", "极轻微稳定动作")
    return not all(fragment in action for fragment in generic_fragments[:2])


def _should_preserve_first_product_shot_contract(shot: dict[str, Any]) -> bool:
    return _shot_has_explicit_material_plan(shot) or _shot_has_detailed_motion_contract(shot)


def _shot_purpose(shot: dict[str, Any]) -> str:
    """兼容新旧分镜字段，避免 LLM 只返回 scene_goal 时创作计划中断。"""

    for key in ("purpose", "scene_goal", "narrative_role", "visual_description"):
        value = str(shot.get(key, "")).strip()
        if value:
            return value
    return "完成当前分镜的商品表达"


def adapt_storyboard_to_render_segments(storyboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把上层分镜时长适配成底层视频模型片段计划。"""

    render_segments: list[dict[str, Any]] = []
    timeline_cursor = 0.0
    model_duration_seconds = 5

    for fallback_index, shot in enumerate(storyboard, start=1):
        shot_index = int(shot.get("shot_index", fallback_index))
        target_duration = _safe_float(shot.get("duration_seconds"), default=5.0)
        target_duration = min(max(target_duration, 1.0), float(model_duration_seconds))

        # Seedance 1.5 按 5 秒生成；这里保留 LLM 目标时长，后处理再裁剪。
        render_segments.append(
            {
                "shot_index": shot_index,
                "target_duration_seconds": target_duration,
                "model_duration_seconds": model_duration_seconds,
                "trim_start_seconds": 0.0,
                "trim_end_seconds": target_duration,
                "timeline_start_seconds": timeline_cursor,
                "timeline_end_seconds": timeline_cursor + target_duration,
            }
        )
        timeline_cursor += target_duration

    return render_segments


def review_rendered_video_content(
    product_context: dict[str, Any],
    creation_plan: dict[str, Any],
    render_result: dict[str, Any],
    output_dir: str,
    shot_indices_filter: list[int] | None = None,
) -> dict[str, Any]:
    """抽帧审视生成视频内容，按分镜维度检查商品主体是否和上传素材一致。
    shot_indices_filter：只审视指定下标的分镜（用于修复后的第2轮验证）。
    """

    if not render_result.get("success") or not render_result.get("video_path"):
        return {
            "passed": False,
            "skipped": False,
            "mode": "content_review",
            "summary": "视频未成功生成，无法进行内容审视。",
            "error": render_result.get("error"),
            "shot_reviews": [],
            "repair_records": [],
        }

    reference_image_paths = _reference_image_paths_for_content_review(product_context, creation_plan)
    if not reference_image_paths:
        return {
            "passed": True,
            "skipped": True,
            "mode": "content_review",
            "summary": "没有可用于商品一致性对比的上传图片，跳过内容审视。",
            "error": None,
            "shot_reviews": [],
            "repair_records": [],
        }

    storyboard = creation_plan.get("shots", [])
    total_duration = float(creation_plan.get("total_duration_seconds", 15))

    frame_records = _extract_video_review_frames(
        video_path=str(render_result["video_path"]),
        output_dir=output_dir,
        duration_seconds=total_duration,
    )
    if not frame_records:
        return {
            "passed": False,
            "skipped": False,
            "mode": "content_review",
            "summary": "无法从生成视频中抽取审视帧。",
            "error": "frame_extraction_failed",
            "shot_reviews": [],
            "repair_records": [],
        }

    frame_records = _bind_frames_to_shots(frame_records, storyboard)

    # 第2轮验证：只审视指定分镜的帧
    if shot_indices_filter is not None:
        frame_records = [r for r in frame_records if r.get("shot_index") in shot_indices_filter]
        storyboard = [s for s in storyboard if s.get("shot_index") in shot_indices_filter]

    prompt = _build_content_review_prompt(product_context, creation_plan, frame_records)
    frame_paths = [record["frame_path"] for record in frame_records if record.get("frame_path")]
    llm_result = _call_multimodal_llm(
        prompt_data=prompt,
        image_paths=reference_image_paths + frame_paths,
        purpose="content_review",
    )
    if not llm_result["ok"]:
        return {
            "passed": False,
            "skipped": False,
            "mode": "content_review",
            "summary": "内容审视模型调用失败，无法确认商品一致性。",
            "error": llm_result.get("error"),
            "shot_reviews": [],
            "repair_records": [
                {
                    "shot_index": shot.get("shot_index"),
                    "action": "rerender_with_image_to_video",
                    "reason": "内容审视不可用，商品主体镜头需要保守绑定上传图。",
                }
                for shot in storyboard
                if (shot.get("asset") or {}).get("file_path")
            ],
        }

    parsed_review = _extract_json_from_text(llm_result["content"])
    review = _normalize_content_review(parsed_review, frame_records, storyboard)
    review["llm_enabled"] = True
    review["llm_notes"] = llm_result["content"]
    return review


def _reference_image_paths_from_creation_plan(creation_plan: dict[str, Any]) -> list[str]:
    """收集创作计划中绑定的上传商品图片，用于和生成视频抽帧做对比。"""

    image_paths: list[str] = []
    for shot in creation_plan.get("shots", []):
        asset = shot.get("asset") or {}
        file_path = str(asset.get("file_path", "")).strip()
        if asset.get("is_scene_background"):
            continue
        if asset.get("asset_type") == "image" and file_path and Path(file_path).exists():
            if file_path not in image_paths:
                image_paths.append(file_path)
    return image_paths


def _reference_image_paths_for_content_review(
    product_context: dict[str, Any],
    creation_plan: dict[str, Any],
) -> list[str]:
    """合并分镜绑定图片和原始上传图片，避免素材未绑定到分镜时跳过审视。"""

    image_paths = _reference_image_paths_from_creation_plan(creation_plan)
    for file_path in product_context.get("reference_image_paths", []):
        normalized_path = str(file_path).strip()
        if normalized_path and Path(normalized_path).exists() and normalized_path not in image_paths:
            image_paths.append(normalized_path)
    return image_paths


def _extract_video_review_frames(
    video_path: str,
    output_dir: str,
    duration_seconds: float = 15.0,
) -> list[dict[str, Any]]:
    """每秒抽一帧，返回帧记录列表。"""

    if not Path(video_path).exists():
        return []

    try:
        import imageio.v2 as imageio
    except ImportError:
        return []

    frame_dir = Path(output_dir) / "content_review_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_records: list[dict[str, Any]] = []

    timestamps = [i + 0.5 for i in range(int(duration_seconds))]

    try:
        reader = imageio.get_reader(video_path)
        metadata = reader.get_meta_data()
        fps = float(metadata.get("fps") or 24)

        for ts in timestamps:
            frame_index = int(ts * fps)
            try:
                frame = reader.get_data(frame_index)
            except Exception:
                continue

            frame_path = frame_dir / f"frame_{ts:05.1f}s.jpg"
            imageio.imwrite(frame_path, frame)
            frame_records.append(
                {
                    "timestamp_seconds": round(ts, 2),
                    "frame_path": str(frame_path),
                    "shot_index": 0,
                }
            )

        reader.close()
    except Exception:
        pass

    return frame_records


def _bind_frames_to_shots(
    frame_records: list[dict[str, Any]],
    storyboard: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """根据时间轴把每张抽帧绑定到分镜，附加分镜审查上下文。"""

    for frame in frame_records:
        ts = frame.get("timestamp_seconds", 0)
        shot = _find_shot_at_time(storyboard, ts)
        if shot:
            frame["shot_index"] = shot.get("shot_index", 0)
            frame["scene_goal"] = shot.get("scene_goal", "")
            frame["product_presence"] = shot.get("product_presence", "optional")
            frame["identity_strictness"] = shot.get("identity_strictness", "low")
            frame["review_focus"] = shot.get("review_focus", [])
            frame["completion_criteria"] = shot.get("completion_criteria", [])
            frame["forbidden_variation"] = shot.get("forbidden_variation", [])
        else:
            frame["shot_index"] = 0
            frame["product_presence"] = "optional"
            frame["identity_strictness"] = "low"
    return frame_records


def _find_shot_at_time(storyboard: list[dict[str, Any]], timestamp: float) -> dict[str, Any] | None:
    """找到时间轴上覆盖给定时间戳的分镜。"""

    elapsed = 0.0
    for shot in storyboard:
        duration = float(shot.get("duration_seconds", 0))
        if elapsed <= timestamp < elapsed + duration:
            return shot
        elapsed += duration
    if storyboard:
        return storyboard[-1]
    return None


def _build_content_review_prompt(
    product_context: dict[str, Any],
    creation_plan: dict[str, Any],
    frame_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """构造内容审视 prompt，按分镜维度审查每帧，绑定分镜审查上下文。"""

    return {
        "task": "对比上传商品参考图和生成视频抽帧，按分镜目标审查每个镜头。",
        "image_order": "先给出所有上传商品参考图，随后给出生成视频抽帧；抽帧按时间顺序排列，每帧标注所属分镜。",
        "review_rules": {
            "reference_images": "参考图是用户上传的真实商品",
            "extracted_frames": "抽帧图是生成视频中的画面",
            "frame_shot_binding": "每张抽帧都属于某个分镜，按分镜维度一起审查",
            "product_required_shot": "如果该分镜 product_presence 为 required，必须严格检查商品类型、主色、结构、品牌区域、关键部件",
            "product_optional_shot": "如果该分镜 product_presence 为 optional/forbidden，不要因为商品缺失判失败，检查场景是否完成铺垫目标",
            "identity_high": "identity_strictness=high 的分镜，商品外观必须和参考图高度一致",
            "identity_low": "identity_strictness=low 的分镜，允许光线和角度轻微变化，但品牌和结构不能变",
            "error_detection": "如果出现错误品牌、乱码文字、水印、商品不合理翻转、形态漂移，判定对应分镜失败",
            "review_focus_rule": "使用 review_focus 确定审查重点",
            "completion_criteria_rule": "使用 completion_criteria 判断分镜是否达成目标",
            "forbidden_variation_rule": "使用 forbidden_variation 判断哪些变化不允许",
        },
        "product_identity_card": product_context.get("product_identity_card", {}),
        "shots": [
            {
                "shot_index": shot.get("shot_index"),
                "scene_goal": shot.get("scene_goal", shot.get("purpose", "")),
                "initial_state": shot.get("initial_state", ""),
                "action": shot.get("action", ""),
                "final_state": shot.get("final_state", ""),
                "subtitle": shot.get("subtitle", ""),
                "product_presence": shot.get("product_presence", "required"),
                "identity_strictness": shot.get("identity_strictness", "medium"),
                "review_focus": shot.get("review_focus", []),
                "completion_criteria": shot.get("completion_criteria", []),
                "forbidden_variation": shot.get("forbidden_variation", []),
            }
            for shot in creation_plan.get("shots", [])
        ],
        "frame_records": frame_records,
        "review_dimensions": [
            "brand_or_logo_consistency",
            "primary_color_consistency",
            "shape_and_component_consistency",
            "action_rationality",
            "shot_goal_not_achieved",
            "wrong_text_or_watermark",
        ],
        "output_format": {
            "passed": False,
            "summary": "整体审视结论",
            "shot_reviews": [
                {
                    "shot_index": 1,
                    "pass": False,
                    "failed_dimensions": ["brand_or_logo_consistency", "shape_and_component_consistency"],
                    "main_issue": "商品主体和上传参考图不一致，商标区域发生变化。",
                    "repair_strategy": "rerender_with_stronger_identity_anchor",
                }
            ],
        },
        "instruction": "只返回 JSON。每个分镜单独判断 pass 或 fail，并给出具体失败维度和修复建议。repair_strategy 可选值：rerender_with_stronger_identity_anchor、simplify_action、rewrite_shot_goal。",
    }


def _map_repair_strategy(failed_dimensions: list[str], shot: dict[str, Any]) -> str:
    """根据失败类型选择修复方式。"""

    if "brand_or_logo_consistency" in failed_dimensions or "shape_and_component_consistency" in failed_dimensions:
        return "rerender_with_stronger_identity_anchor"
    if "action_rationality" in failed_dimensions:
        return "simplify_action"
    if "shot_goal_not_achieved" in failed_dimensions:
        return "rewrite_shot_goal"
    return "rerender_with_stronger_identity_anchor"


def _normalize_content_review(
    raw_review: Any,
    frame_records: list[dict[str, Any]],
    storyboard: list[dict[str, Any]],
) -> dict[str, Any]:
    """归一化内容审视结果，按分镜生成修复建议。"""

    if not isinstance(raw_review, dict):
        return {
            "passed": False,
            "skipped": False,
            "mode": "multimodal_frame_review",
            "summary": "内容审视结果不是合法 JSON。",
            "error": "invalid_content_review_json",
            "shot_reviews": [],
            "repair_records": [],
            "frame_records": frame_records,
        }

    shot_reviews = raw_review.get("shot_reviews", [])
    if not isinstance(shot_reviews, list):
        shot_reviews = []

    shots_by_index = {shot.get("shot_index"): shot for shot in storyboard}

    normalized_reviews = []
    for item in shot_reviews:
        if not isinstance(item, dict):
            continue
        failed_dimensions = _string_list(item.get("failed_dimensions", []))
        shot_index = item.get("shot_index")
        shot = shots_by_index.get(shot_index, {})
        repair_strategy = str(item.get("repair_strategy", "")).strip()
        if not repair_strategy:
            repair_strategy = _map_repair_strategy(failed_dimensions, shot)
        normalized_reviews.append(
            {
                "shot_index": shot_index,
                "pass": bool(item.get("pass", item.get("passed", False))),
                "failed_dimensions": failed_dimensions,
                "main_issue": str(item.get("main_issue", "")).strip(),
                "repair_strategy": repair_strategy,
            }
        )

    passed = bool(raw_review.get("passed", all(item.get("pass") for item in normalized_reviews)))
    repair_records = _build_content_repair_records(normalized_reviews)
    return {
        "passed": passed and not repair_records,
        "skipped": False,
        "mode": "multimodal_frame_review",
        "summary": str(raw_review.get("summary", "")).strip() or ("通过" if passed else "内容审视未通过"),
        "error": None,
        "shot_reviews": normalized_reviews,
        "repair_records": repair_records,
        "frame_records": frame_records,
    }


def _build_content_repair_records(shot_reviews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把失败的审视项转成可执行的局部修复策略。"""

    repair_records: list[dict[str, Any]] = []
    for item in shot_reviews:
        if item.get("pass"):
            continue
        failed_dimensions = set(item.get("failed_dimensions", []))
        if "product_consistency" in failed_dimensions or "brand_or_logo_consistency" in failed_dimensions:
            # Logo 或主体已经漂移时，继续生成只会再次重绘。改用上传素材生成确定性保真片段。
            action = "fallback_to_local_identity_anchor"
        elif "action_completion" in failed_dimensions:
            action = "simplify_action_and_rerender"
        elif "shape_and_component_consistency" in failed_dimensions:
            action = "rerender_with_stronger_identity_anchor"
        else:
            action = "fallback_to_local_ken_burns"

        repair_records.append(
            {
                "shot_index": item.get("shot_index"),
                "action": action,
                "reason": item.get("main_issue", "内容审视未通过"),
                "repair_strategy": (
                    action
                    if action == "fallback_to_local_identity_anchor"
                    else item.get("repair_strategy", action)
                ),
            }
        )
    return repair_records


def run_final_check(
    product_context: dict[str, Any],
    storyboard: list[dict[str, Any]],
    creation_plan: dict[str, Any],
    render_result: dict[str, Any],
    asset_gap_completion: dict[str, Any] | None = None,
    content_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """检查草稿计划是否满足当前 MVP 的基本要求。"""

    print("[video_generation_workflow] 开始最终检查。", flush=True)
    from agent.final_checks import run_final_check as _run_final_check

    result = _run_final_check(
        product_context=product_context,
        storyboard=storyboard,
        creation_plan=creation_plan,
        render_result=render_result,
        asset_gap_completion=asset_gap_completion,
        content_review=content_review,
    )
    print(f"[video_generation_workflow] 最终检查完成：passed={result['passed']}", flush=True)
    return result


def _build_asset_profiles(assets: list[dict[str, Any]], semantic_summary: str) -> list[dict[str, Any]]:
    """把上传素材整理成素材画像；先用规则兜底，后续可替换为更细的多模态结构化输出。"""

    profiles: list[dict[str, Any]] = []
    for index, asset in enumerate(assets, start=1):
        asset_id = asset.get("asset_id") or f"asset_{index:03d}"
        role = "appearance_anchor" if asset.get("asset_type") == "image" else "direct_clip"
        if not asset.get("is_supported"):
            role = "negative_reference"

        # 这里不从文件名臆测商品细节，只记录素材在生成链路里的可用角色。
        profile = {
            "asset_id": asset_id,
            "filename": asset.get("filename", ""),
            "asset_type": asset.get("asset_type", "unknown"),
            "visual_role": role,
            "quality_score": 80 if asset.get("is_supported") else 20,
            "product_visibility": "unknown",
            "detail_visibility": "unknown",
            "background_type": "unknown",
            "scene_type": "unknown",
            "suitable_for": _asset_profile_suitable_for(role),
            "not_suitable_for": [] if asset.get("is_supported") else ["video_generation"],
            "identity_contribution": ["商品外观约束"] if role == "appearance_anchor" else [],
            "risk_notes": [] if semantic_summary else ["当前素材语义理解较弱，仅使用基础角色判断。"],
            "role_source": "fallback",
        }
        _normalize_material_roles(profile)
        profiles.append(profile)
    return profiles


def _normalize_material_roles(profile: dict[str, Any]) -> dict[str, Any]:
    """保留多模态原始 visual_role，同时派生可供策略层使用的素材能力标签。"""

    visual_role = str(profile.get("visual_role", "")).strip()
    suitable_for = _string_list(profile.get("suitable_for", []))
    product_visibility = str(profile.get("product_visibility", "")).strip()
    normalized_roles = _unique_list(
        _string_list(profile.get("normalized_roles", []))
        + ([visual_role] if visual_role else [])
        + suitable_for
    )
    material_capabilities = _safe_dict(profile.get("material_capabilities"))

    has_clear_product = any(
        word in product_visibility
        for word in ("主体清晰", "完整清晰", "商品清晰", "清晰可见", "完整可见")
    )
    if visual_role in {"appearance_anchor", "full_product_anchor"}:
        normalized_roles.append("appearance_anchor_candidate")
        material_capabilities["appearance_anchor_candidate"] = True
    if (
        visual_role == "scene_context"
        and has_clear_product
        and any(role in suitable_for for role in ("product_showcase", "product_reveal", "product_hero"))
    ):
        normalized_roles.append("appearance_anchor_candidate")
        material_capabilities["appearance_anchor_candidate"] = True
        material_capabilities["candidate_reason"] = (
            "素材原始角色是 scene_context，但主体清晰且 suitable_for 包含商品展示职责，"
            "可作为商品确认/首帧锚点候选。"
        )
    if visual_role in {"detail_reference", "logo_detail", "brand_detail"}:
        material_capabilities["detail_reference"] = True
    if visual_role == "scene_context":
        material_capabilities["scene_context"] = True

    profile["normalized_roles"] = _unique_list(normalized_roles)
    profile["material_capabilities"] = material_capabilities
    return profile


def _build_asset_profiles_from_parsed(
    asset_roles: list[dict[str, Any]],
    assets: list[dict[str, Any]],
    role_source: str = "multimodal",
) -> list[dict[str, Any]]:
    """从 LLM 解析后的 asset_roles 数组构建素材画像。"""

    if not isinstance(asset_roles, list) or not asset_roles:
        return _build_asset_profiles(assets, "")

    asset_map = {a.get("asset_id", ""): a for a in assets}
    profiles: list[dict[str, Any]] = []
    for role_entry in asset_roles:
        if not isinstance(role_entry, dict):
            continue
        asset_id = str(role_entry.get("asset_id", "")).strip()
        asset = asset_map.get(asset_id, {})
        if not asset:
            continue
        profile = {
            "asset_id": asset_id,
            "filename": asset.get("filename", ""),
            "asset_type": asset.get("asset_type", "unknown"),
            "visual_role": str(role_entry.get("visual_role", "appearance_anchor")),
            "quality_score": int(role_entry.get("quality_score", 80)),
            "product_visibility": str(role_entry.get("product_visibility", "unknown")),
            "detail_visibility": str(role_entry.get("detail_visibility", "unknown")),
            "background_type": str(role_entry.get("background_type", "unknown")),
            "scene_type": str(role_entry.get("scene_type", "unknown")),
            "suitable_for": [str(r) for r in role_entry.get("suitable_for", [])],
            "not_suitable_for": [str(r) for r in role_entry.get("not_suitable_for", [])],
            "identity_contribution": [str(c) for c in role_entry.get("identity_contribution", [])],
            "risk_notes": [str(n) for n in role_entry.get("risk_notes", [])],
            "role_source": role_source,
        }
        _normalize_material_roles(profile)
        profiles.append(profile)
    return profiles or _build_asset_profiles(assets, "")


def _format_role_summary_from_parsed(
    asset_roles: list[dict[str, Any]],
    assets: list[dict[str, Any]],
) -> str:
    """从 LLM 解析后的 asset_roles 数组生成可读的语义摘要文本。"""

    if not isinstance(asset_roles, list) or not asset_roles:
        return "素材分析结果不可用，使用规则兜底。"
    lines: list[str] = []
    for entry in asset_roles:
        if not isinstance(entry, dict):
            continue
        asset_id = entry.get("asset_id", "")
        suitable = entry.get("suitable_for", [])
        reason = entry.get("reason", "")
        line = f"### {asset_id}\n适合角色：{', '.join(suitable) if suitable else '未确定'}"
        if reason:
            line += f"\n理由：{reason}"
        lines.append(line)
    return "\n\n".join(lines) if lines else "素材分析结果不可用。"


def _assets_for_llm(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """生成给 LLM 使用的素材元数据，刻意移除文件名、本地路径和访问链接。"""

    safe_assets: list[dict[str, Any]] = []
    for asset in assets:
        safe_assets.append(
            {
                "asset_id": asset.get("asset_id", ""),
                "asset_type": asset.get("asset_type", "unknown"),
                "content_type": asset.get("content_type", ""),
                "file_size": asset.get("file_size", 0),
                "is_supported": asset.get("is_supported", False),
                "suggested_role": asset.get("suggested_role", ""),
            }
        )
    return safe_assets


def _asset_profiles_for_llm(asset_profiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """清洗素材画像，避免模型从文件名推断商品事实。"""

    safe_profiles: list[dict[str, Any]] = []
    for profile in asset_profiles:
        safe_profile = dict(profile)
        safe_profile.pop("filename", None)
        safe_profiles.append(safe_profile)
    return safe_profiles


def _asset_analysis_for_llm(asset_analysis: dict[str, Any]) -> dict[str, Any]:
    """清洗完整素材分析结果，避免后续模型从文件名或本地路径推断视觉事实。"""

    safe_analysis = dict(asset_analysis)
    safe_analysis["assets"] = _assets_for_llm(asset_analysis.get("assets", []))
    safe_analysis["asset_profiles"] = _asset_profiles_for_llm(asset_analysis.get("asset_profiles", []))
    return safe_analysis



def _image_paths_from_asset_analysis(asset_analysis: dict[str, Any]) -> list[str]:
    """从素材分析结果中提取可直接发送给多模态模型的本地图片路径。"""

    image_paths: list[str] = []
    for asset in asset_analysis.get("assets", []):
        file_path = str(asset.get("file_path", "")).strip()
        if asset.get("asset_type") == "image" and file_path and Path(file_path).exists():
            image_paths.append(file_path)
    return image_paths


def _product_context_for_llm(product_context: dict[str, Any]) -> dict[str, Any]:
    """清洗商品上下文，保留创作所需字段，但不向模型暴露素材文件名。"""

    safe_context = dict(product_context)
    safe_context["asset_profiles"] = _asset_profiles_for_llm(product_context.get("asset_profiles", []))
    return safe_context


def _asset_metadata_for_llm(assets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """给导演节点看的素材清单，只暴露稳定 ID 和类型，不暴露文件名或路径。"""

    metadata: list[dict[str, Any]] = []
    for asset in assets:
        metadata.append(
            {
                "asset_id": str(asset.get("asset_id", "")).strip(),
                "asset_type": str(asset.get("asset_type", "")).strip(),
                "is_supported": bool(asset.get("is_supported")),
                "suggested_role": str(asset.get("suggested_role", "")).strip(),
            }
        )
    return metadata


def _apply_conservative_strategy(
    product_context: dict[str, Any],
    product_identity_card: dict[str, Any],
) -> dict[str, Any]:
    constraints = {
        "can_create_freely": ["办公桌面场景", "通勤携带场景", "明亮或高级感风格", "镜头推近、横移、定镜等安全运镜"],
        "cannot_create_freely": ["改变商品品牌标识", "改变机身颜色", "改变屏幕和键盘布局", "让笔记本不合理翻转、悬浮、变形"],
        "strategy": "conservative",
        "preferred_render_mode": "image_to_video_with_anchor",
        "avoid_complex_motion": True,
        "prefer_ken_burns": True,
    }
    existing_preserve = set(product_identity_card.get("must_preserve", []))
    for item in constraints["cannot_create_freely"]:
        existing_preserve.add(item)
    constraints["must_preserve"] = list(existing_preserve)
    return constraints


def _normalize_product_identity_card(
    raw_card: Any,
    task_data: dict[str, Any],
    asset_analysis: dict[str, Any],
) -> dict[str, Any] | None:
    """归一化 LLM 返回的商品身份卡，保证关键字段始终存在。"""

    if not isinstance(raw_card, dict):
        return None

    fallback = _fallback_product_identity_card(task_data, asset_analysis)
    card = dict(fallback)
    for key in [
        "product_type",
        "identity_confidence",
        "appearance_summary",
        "primary_color",
    ]:
        value = str(raw_card.get(key, "")).strip()
        if value:
            card[key] = value

    for key in [
        "secondary_colors",
        "material_features",
        "shape_features",
        "key_components",
        "visible_marks",
        "functional_features",
        "scale_or_size_cues",
        "must_preserve",
        "allowed_variations",
        "forbidden_changes",
        "reference_asset_ids",
    ]:
        value = raw_card.get(key)
        if isinstance(value, list):
            card[key] = [str(item).strip() for item in value if str(item).strip()]

    if isinstance(raw_card.get("motion_affordance"), dict):
        card["motion_affordance"] = _normalize_motion_affordance(
            raw_card["motion_affordance"],
            card["product_type"],
        )
    return card


def build_product_identity_card(
    task_data: dict[str, Any],
    asset_analysis: dict[str, Any],
    structured_requirements: dict[str, Any] | list[str] | None = None,
) -> dict[str, Any]:
    """兼容旧调用入口：复用素材处理产物，不再次调用多模态模型。"""

    task_with_requirements = dict(task_data)
    if isinstance(structured_requirements, dict):
        task_with_requirements["structured_requirements"] = structured_requirements
    raw_card = asset_analysis.get("product_identity_card", {})
    normalized = _normalize_product_identity_card(raw_card, task_with_requirements, asset_analysis)
    return normalized or _fallback_product_identity_card(task_with_requirements, asset_analysis)


def _fallback_product_identity_card(
    task_data: dict[str, Any],
    asset_analysis: dict[str, Any],
) -> dict[str, Any]:
    """LLM 不可用时，根据标题、卖点和素材画像生成保守的商品身份卡。"""

    title = str(task_data.get("title", "")).strip()
    product_type = _infer_product_type(title)
    primary_color = _infer_primary_color(title)
    selling_points = [str(point).strip() for point in task_data.get("selling_points", []) if str(point).strip()]
    reference_asset_ids = [
        profile.get("asset_id")
        for profile in asset_analysis.get("asset_profiles", [])
        if profile.get("visual_role") == "appearance_anchor"
    ]

    card = {
        "product_type": product_type,
        # 没有多模态视觉事实时，即使标题存在也只能作为低置信度文字线索。
        "identity_confidence": "low",
        "appearance_summary": _fallback_appearance_summary(title, product_type, primary_color),
        "primary_color": primary_color,
        "secondary_colors": [],
        "material_features": [],
        "shape_features": _fallback_shape_features(product_type),
        "key_components": _fallback_key_components(product_type),
        "visible_marks": [],
        "functional_features": selling_points,
        "scale_or_size_cues": [],
        "must_preserve": [
            f"保持{product_type}商品类型",
            "保持商品主体形态和关键结构",
            f"保持{primary_color}主色调" if primary_color else "保持素材中的主色调",
        ],
        "allowed_variations": ["背景", "光照", "摆放角度", "轻微镜头运动"],
        "forbidden_changes": [
            "不能把商品变成其他品类",
            "不能改变商品主色调",
            "不能增加素材中不存在的品牌文字或 logo",
            "不能让商品发生不符合真实结构的变形",
        ],
        "reference_asset_ids": reference_asset_ids,
        "motion_affordance": _fallback_motion_affordance(product_type),
    }
    structured_req = _safe_dict(task_data.get("structured_requirements", {}))
    user_must_preserve = structured_req.get("must_preserve", [])
    user_avoid = structured_req.get("avoid", [])
    if user_must_preserve:
        existing_must = set(card.get("must_preserve", []))
        card["must_preserve"] = list(existing_must | set(user_must_preserve))
    if user_avoid:
        existing_forbidden = set(card.get("forbidden_changes", []))
        card["forbidden_changes"] = list(existing_forbidden | set(user_avoid))
    return card


def _normalize_motion_affordance(raw_motion: dict[str, Any], product_type: str) -> dict[str, Any]:
    """把模型输出的动作能力补齐默认值，避免后续 prompt 缺字段。"""

    fallback = _fallback_motion_affordance(product_type)
    motion = dict(fallback)
    for key in [
        "can_move_by_itself",
        "can_fly",
        "can_open_or_close",
        "can_fold_or_transform",
        "can_be_worn",
        "can_be_handheld",
        "can_be_used_by_human",
        "requires_human_interaction",
    ]:
        if key in raw_motion:
            motion[key] = bool(raw_motion[key])

    if raw_motion.get("can_rotate"):
        motion["can_rotate"] = str(raw_motion["can_rotate"])
    for key in ["allowed_actions", "risky_actions", "forbidden_actions"]:
        if isinstance(raw_motion.get(key), list):
            motion[key] = [str(item).strip() for item in raw_motion[key] if str(item).strip()]
    return motion


def _fallback_motion_affordance(product_type: str) -> dict[str, Any]:
    """根据商品类型生成保守的动作能力边界。"""

    can_fly = product_type in {"无人机", "飞行玩具"}
    can_move = product_type in {"玩具车", "扫地机器人", "无人机", "飞行玩具"}
    can_wear = product_type in {"服饰", "鞋", "饰品"}
    can_fold = product_type in {"折叠桌", "折叠椅", "笔记本电脑"}

    forbidden_actions = []
    if not can_fly:
        forbidden_actions.append("飞行")
    if not can_fold:
        forbidden_actions.append("展开变形")
    if not can_move:
        forbidden_actions.append("自动跳跃")

    return {
        "can_move_by_itself": can_move,
        "can_fly": can_fly,
        "can_rotate": "camera_orbit_only",
        "can_open_or_close": product_type in {"盒子", "笔记本电脑"},
        "can_fold_or_transform": can_fold,
        "can_be_worn": can_wear,
        "can_be_handheld": not can_wear,
        "can_be_used_by_human": True,
        "requires_human_interaction": False,
        "allowed_actions": ["镜头推近", "镜头环绕", "手持展示", "桌面滑动"],
        "risky_actions": ["快速旋转", "复杂多人互动"],
        "forbidden_actions": forbidden_actions,
    }


def _asset_profile_suitable_for(role: str) -> list[str]:
    """根据素材角色给出适用镜头任务。"""

    mapping = {
        "appearance_anchor": ["product_reveal", "feature_demo", "detail_proof"],
        "direct_clip": ["usage_demo", "scene_context"],
        "negative_reference": [],
    }
    return mapping.get(role, ["style_reference"])


def _infer_product_type(title: str) -> str:
    """从标题中提取粗粒度商品类型；只做保守关键词匹配。"""

    normalized_title = title.lower()
    if "笔记本" in title and any(
        keyword in normalized_title
        for keyword in (
            "笔记本电脑",
            "电脑",
            "laptop",
            "gaming notebook",
            "游戏本",
            "轻薄本",
            "电竞",
            "雷蛇",
            "razer",
        )
    ):
        return "笔记本电脑"

    product_keywords = [
        "鼠标",
        "键盘",
        "耳机",
        "音箱",
        "显示器",
        "无人机",
        "飞行玩具",
        "玩具车",
        "扫地机器人",
        "折叠桌",
        "折叠椅",
        "笔记本电脑",
        "鞋",
        "服饰",
        "饰品",
    ]
    for keyword in product_keywords:
        if keyword in title:
            return keyword
    return title or "商品"


def _infer_primary_color(title: str) -> str:
    """从标题中提取主色调，提取不到时保持 unknown。"""

    for color in ["白色", "黑色", "银色", "灰色", "红色", "蓝色", "绿色", "粉色", "金色"]:
        if color in title:
            return color
    return "unknown"


def _fallback_shape_features(product_type: str) -> list[str]:
    """给常见商品补一个保守外形描述，避免 prompt 完全没有结构约束。"""

    if product_type == "鼠标":
        return ["扁平椭圆轮廓", "左右按键", "滚轮"]
    if product_type == "键盘":
        return ["矩形键盘主体", "多排按键"]
    return [f"{product_type}常见轮廓"]


def _fallback_key_components(product_type: str) -> list[str]:
    """给常见商品补关键部件，用于后续商品一致性约束。"""

    if product_type == "鼠标":
        return ["左键", "右键", "滚轮", "鼠标外壳"]
    if product_type == "键盘":
        return ["键帽", "键盘外壳", "按键区域"]
    return [f"{product_type}主体"]


def _fallback_appearance_summary(title: str, product_type: str, primary_color: str) -> str:
    """生成身份卡里的稳定外观摘要。"""

    color_text = "" if primary_color == "unknown" else f"{primary_color}"
    if title:
        return f"{title}，{color_text}{product_type}，保持素材中的外观、颜色和关键结构。"
    return f"{color_text}{product_type}，保持素材中的外观、颜色和关键结构。"


def _normalize_asset(asset: dict[str, Any]) -> dict[str, Any]:
    """把任务里的上传素材转换成工作流内部使用的素材记录。"""

    filename = str(asset.get("filename", "unnamed"))
    content_type = str(asset.get("content_type", "application/octet-stream"))
    asset_type = str(asset.get("asset_type") or "").strip() or _detect_asset_type(content_type)
    if asset_type == "unknown":
        file_path = str(asset.get("file_path", ""))
        guessed_type = mimetypes.guess_type(file_path)[0] or ""
        asset_type = _detect_asset_type(guessed_type)
    is_supported = asset_type in {"image", "video"}
    stable_asset_id = hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]
    return {
        # asset_id 后续会被商品身份卡、素材画像和分镜引用，必须跨进程保持稳定。
        "asset_id": str(asset.get("asset_id", "")) or f"asset_{stable_asset_id}",
        "filename": filename,
        "content_type": content_type,
        "file_path": str(asset.get("file_path", "")),
        "original_file_path": str(asset.get("file_path", "")),
        "public_url": str(asset.get("public_url", "")),
        "source_url": str(asset.get("source_url", "")),
        "file_size": int(asset.get("file_size", 0) or 0),
        "primary_product": _safe_dict(asset.get("primary_product", {})),
        "asset_type": asset_type,
        "is_supported": is_supported,
        "suggested_role": _suggest_asset_role(asset_type),
    }


def _load_local_env() -> None:
    """
    加载本地 `.env` 文件。

    这里不引入 `python-dotenv`，避免为了一个开发期能力增加依赖。
    已存在的系统环境变量优先级更高，不会被 `.env` 覆盖。
    """

    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value

    _pin_dns_for_wsl2()


def _pin_dns_for_wsl2() -> None:
    """
    WSL2 环境下 DNS 解析可能间歇性很慢（每次 5-19 秒）。
    对已知 API 域名做 DNS pinning，绕过 getaddrinfo 查询，直接返回 IP。
    """

    try:
        _setup_dns_pins()
    except Exception:
        pass


def _setup_dns_pins() -> None:  # pragma: no cover — 依赖具体 IP 和 WSL2 环境
    """对已知 API 域名注入硬编码 IP，避免每次请求都走慢 DNS。"""

    import socket

    _PINNED: dict[str, list[str]] = {
        "ark.cn-beijing.volces.com": ["101.126.13.31"],
        "api.deepseek.com": ["111.32.200.78"],
    }

    _original = socket.getaddrinfo

    def _patched(host, port, family=0, type=0, proto=0, flags=0):
        if isinstance(host, str) and host in _PINNED:
            results: list[tuple[Any, ...]] = []
            for ip in _PINNED[host]:
                results.append((socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port)))
            return results
        return _original(host, port, family, type, proto, flags)

    socket.getaddrinfo = _patched


def _merge_style_text(product_context: dict[str, Any]) -> str:
    """合并风格预设和用户自定义风格期望。"""

    style = str(product_context.get("style", "")).strip()
    custom_style_prompt = str(product_context.get("custom_style_prompt", "")).strip()
    if style and custom_style_prompt:
        return f"{style}；用户补充：{custom_style_prompt}"
    return custom_style_prompt or style or "清晰直接的商品展示风格"


def _run_step_with_review(
    step_name: str,
    generate_func,
    review_func,
    max_retries: int,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    """执行"生成 -> 审核 -> 必要时打回重做"的受控循环。"""

    attempts: list[dict[str, Any]] = []
    previous_issues: list[str] = []
    last_result: Any = None
    last_review: dict[str, Any] = _review_result(
        passed=False,
        issues=["节点尚未执行。"],
        retry_target=step_name,
        retryable=True,
    )

    for attempt_index in range(1, max_retries + 2):
        print(
            "[video_generation_workflow] 执行可打回节点："
            f"step={step_name}, attempt={attempt_index}",
            flush=True,
        )

        # 把上一次审核发现的问题传回生成函数。
        # 这样重试不是简单重复，而是带着明确修改意见重新生成。
        last_result = generate_func(previous_issues)
        last_review = review_func(last_result)

        action = "continue" if last_review["passed"] else "retry"
        if not last_review["retryable"]:
            action = "stop"
        if attempt_index > max_retries and not last_review["passed"]:
            action = "needs_user_review"

        attempts.append(
            {
                "step": step_name,
                "attempt": attempt_index,
                "passed": last_review["passed"],
                "issues": last_review["issues"],
                "action": action,
            }
        )

        if last_review["passed"] or not last_review["retryable"]:
            break

        previous_issues = last_review["issues"]

    return last_result, last_review, attempts


def _review_result(
    passed: bool,
    issues: list[str],
    retry_target: str | None,
    retryable: bool,
) -> dict[str, Any]:
    """生成统一的审核结果。"""

    if passed:
        severity = "ok"
    elif retryable:
        severity = "warning"
    else:
        severity = "error"

    return {
        "passed": passed,
        "severity": severity,
        "issues": issues,
        "retryable": retryable,
        "retry_target": retry_target,
    }


def _review_step_status(review: dict[str, Any]) -> str:
    """把审核结果转换成页面可展示的步骤状态。"""

    return "completed" if review.get("passed") else "needs_review"


def _select_auto_repair_records(
    repair_records: list[dict[str, Any]],
    failed_count: int,
    total_shots: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """自动修复只处理少量局部问题；系统性失败交给人工复核。"""

    total_shots = max(1, int(total_shots or 1))
    if failed_count >= max(3, (total_shots // 2) + 1):
        return [], {
            "auto_repair_enabled": False,
            "reason": (
                f"内容审视失败 {failed_count}/{total_shots} 个镜头，属于系统性计划失败；"
                "已停止自动重渲染，避免长时间反复修复。"
            ),
            "max_auto_repairs": 0,
            "selected_count": 0,
            "original_count": len(repair_records),
        }

    max_auto_repairs = int(os.getenv("AIGC_MAX_AUTO_CONTENT_REPAIRS", "2") or "2")
    max_auto_repairs = max(0, max_auto_repairs)
    selected = list(repair_records[:max_auto_repairs])
    return selected, {
        "auto_repair_enabled": bool(selected),
        "reason": f"仅自动修复前 {len(selected)} 个局部失败镜头，剩余问题保留给人工复核。",
        "max_auto_repairs": max_auto_repairs,
        "selected_count": len(selected),
        "original_count": len(repair_records),
    }


def _skipped_repair_execution(policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempted_count": 0,
        "succeeded_count": 0,
        "failed_count": 0,
        "skipped_count": int(policy.get("original_count", 0) or 0),
        "reconcat_success": False,
        "repair_records": [],
        "records": [],
        "skipped_reason": policy.get("reason", "auto_repair_disabled"),
    }


def _repair_rendered_content(
    task_id: str,
    repair_records: list[dict[str, Any]],
    creation_plan: dict[str, Any],
    render_result: dict[str, Any],
    output_dir: str,
    report: Callable[..., None],
) -> dict[str, Any]:
    """根据内容审视的修复建议，对失败分镜进行自动修复重渲染。"""

    from agent.content_repair import repair_rendered_content

    return repair_rendered_content(
        task_id=task_id,
        repair_records=repair_records,
        creation_plan=creation_plan,
        render_result=render_result,
        output_dir=output_dir,
        report=report,
        repair_func=repair_and_rerender_shot,
        flow_print=_flow_print,
    )


def _build_trace_summary(
    asset_analysis: dict[str, Any],
    script_plan: dict[str, Any],
    review_attempts: list[dict[str, Any]],
    render_result: dict[str, Any],
    final_check: dict[str, Any],
    asset_gap_completion: dict[str, Any] | None = None,
    content_review: dict[str, Any] | None = None,
    storyboard: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """聚合一份轻量 trace 摘要，方便页面解释工作流执行路径。"""

    # 导演+分镜合并节点：LLM 调用尝试过即标记为已使用。
    _sb = storyboard or []
    _director_llm = bool(_sb and any(s.get("render_strategy") for s in _sb))

    return {
        "llm_usage": {
            "asset_analysis": bool(asset_analysis.get("llm_enabled")),
            "script_plan": bool(script_plan.get("llm_enabled")),
            "director_storyboard": _director_llm,
            "content_review": bool((content_review or {}).get("llm_enabled")),
        },
        "llm_errors": {
            "asset_analysis": asset_analysis.get("llm_error"),
        },
        "review_attempt_count": review_attempts if isinstance(review_attempts, int) else len(review_attempts),
        "retry_count": max(0, (review_attempts if isinstance(review_attempts, int) else len(review_attempts)) - 1),
        "render_mode": render_result.get("render_mode", "unknown"),
        "fallback_used": bool(render_result.get("fallback_from")),
        "seedance_shot_count": len(render_result.get("shot_results", [])),
        "subtitle_overlay_success": bool((render_result.get("subtitle_overlay") or {}).get("success")),
        "asset_gap_count": len((asset_gap_completion or {}).get("gap_records", [])),
        "asset_gap_unresolved_count": int((asset_gap_completion or {}).get("unresolved_count", 0) or 0),
        "asset_gap_risk_count": int((asset_gap_completion or {}).get("risk_count", 0) or 0),
        "content_review_passed": bool((content_review or {}).get("passed", True)),
        "content_review_skipped": bool((content_review or {}).get("skipped", False)),
        "content_repair_count": len((content_review or {}).get("repair_records", [])),
        "content_repair_succeeded_count": int(
            ((content_review or {}).get("repair_execution") or {}).get("succeeded_count", 0) or 0
        ),
        "content_repair_failed_count": int(
            ((content_review or {}).get("repair_execution") or {}).get("failed_count", 0) or 0
        ),
        "final_check_passed": bool(final_check.get("passed")),
        "final_issue_count": len(final_check.get("issues", [])),
    }


def _extract_json_from_text(text: str) -> Any:
    """从 LLM 文本中提取 JSON 对象或数组。"""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # 有些模型会在 JSON 前后加解释文本，这里只截取最外层 JSON。
    first_object = cleaned.find("{")
    first_array = cleaned.find("[")
    candidates = [index for index in [first_object, first_array] if index >= 0]
    if not candidates:
        return None

    start = min(candidates)
    end_char = "}" if cleaned[start] == "{" else "]"
    end = cleaned.rfind(end_char)
    if end <= start:
        return None

    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None


def _repair_json_response(raw_text: str, purpose: str, expected_shape: Any | None = None) -> dict[str, Any]:
    """把模型已返回的文本修成合法 JSON；不补充新的视觉事实。"""

    raw_text = str(raw_text or "").strip()
    if not raw_text:
        return {"ok": False, "parsed": None, "content": "", "error": "empty_response"}

    prompt = {
        "task": "把下面模型返回内容转换成合法 JSON。",
        "purpose": purpose,
        "hard_rules": [
            "只能整理原文中已经出现的信息，不能添加新的图片理解或商品事实。",
            "如果原文没有足够信息，返回空结构，但仍必须是合法 JSON。",
            "不要返回解释文字。",
        ],
        "expected_shape": expected_shape or {},
        "raw_response": raw_text,
    }
    llm_result = _call_text_llm(prompt, purpose=f"{purpose}_json_repair", temperature=0.0)
    if not llm_result.get("ok"):
        return {
            "ok": False,
            "parsed": None,
            "content": llm_result.get("content", ""),
            "error": llm_result.get("error") or "json_repair_llm_failed",
        }
    parsed = _extract_json_from_text(str(llm_result.get("content", "")))
    if parsed is None:
        return {
            "ok": False,
            "parsed": None,
            "content": llm_result.get("content", ""),
            "error": "json_repair_parse_failed",
        }
    return {
        "ok": True,
        "parsed": parsed,
        "content": llm_result.get("content", ""),
        "error": None,
    }


def _build_asset_selection_diagnostics(
    *,
    assets: list[dict[str, Any]],
    asset_profiles: list[dict[str, Any]],
    fallback_used: bool,
    fallback_reason: str,
    vision_parse_failed: bool,
    vision_parse_repaired: bool,
) -> list[dict[str, Any]]:
    """记录素材画像来源和派生能力，帮助解释后续为什么用/不用某张图。"""

    profiles_by_id = {str(profile.get("asset_id", "")): profile for profile in asset_profiles}
    diagnostics: list[dict[str, Any]] = []
    for asset in assets:
        asset_id = str(asset.get("asset_id", ""))
        profile = profiles_by_id.get(asset_id, {})
        visual_role = str(profile.get("visual_role") or asset.get("visual_role") or "").strip()
        suitable_for = _string_list(profile.get("suitable_for") or asset.get("suitable_for") or [])
        normalized_roles = _string_list(profile.get("normalized_roles") or asset.get("normalized_roles") or [])
        capabilities = _safe_dict(profile.get("material_capabilities") or asset.get("material_capabilities"))
        role_source = str(profile.get("role_source") or ("fallback" if fallback_used else "multimodal")).strip()
        reason_parts = []
        if fallback_used:
            reason_parts.append(f"多模态不可用或解析失败，使用兜底画像：{fallback_reason}")
        else:
            reason_parts.append("来自多模态素材分析。")
        if vision_parse_repaired:
            reason_parts.append("原始返回曾解析失败，已通过文本 JSON 修复保留结构化结果。")
        if capabilities.get("appearance_anchor_candidate"):
            reason_parts.append(str(capabilities.get("candidate_reason") or "派生为 appearance_anchor_candidate。"))
        elif visual_role:
            reason_parts.append(f"原始 visual_role={visual_role}。")
        diagnostics.append(
            {
                "asset_id": asset_id,
                "filename": asset.get("filename", ""),
                "visual_role": visual_role,
                "suitable_for": suitable_for,
                "product_visibility": profile.get("product_visibility", asset.get("product_visibility", "unknown")),
                "normalized_roles": normalized_roles,
                "material_capabilities": capabilities,
                "role_source": role_source,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "vision_parse_failed": vision_parse_failed,
                "vision_parse_repaired": vision_parse_repaired,
                "why_candidate": "；".join(part for part in reason_parts if part),
                "why_used_or_not_used": "最终是否使用取决于分镜职责、素材质量、风险评分和 match_assets_to_storyboard 的逐镜匹配结果。",
            }
        )
    return diagnostics



def _format_creative_direction(director_decision: dict[str, Any]) -> str:
    """把导演因子组合格式化成下游 creative prompt 可以直接引用的创意方向文本。"""

    combo = director_decision.get("factor_combination", {}) or {}
    if not combo:
        return str(director_decision.get("selected_strategy", "标准商品展示"))

    dim_names = {
        "narrative_framework": "叙事框架", "hook": "开场方式", "pacing": "节奏",
        "camera": "运镜", "visual_focus": "画面重心", "exit": "退场", "emotion": "情绪基调",
    }
    parts = [f"{dim_names[k]}={v}" for k, v in combo.items() if v and k in dim_names]
    return "；".join(parts) if parts else str(director_decision.get("selected_strategy", "标准商品展示"))


def _safe_score(value: Any) -> int:
    """把模型评分限制在 0-100，避免页面展示异常。"""

    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return 0
    return max(0, min(100, score))


def _fallback_conservative_script(product_context: dict[str, Any], director_decision: dict[str, Any]) -> dict[str, Any]:
    duration = max(8, int(product_context.get("duration_seconds", 15)))
    selling_points = _product_context_selling_points(product_context)
    first_point = selling_points[0] if selling_points else _fallback_public_caption(product_context, "feature_demo", scene_goal="核心卖点看清")
    second_point = selling_points[1] if len(selling_points) > 1 else _fallback_public_caption(product_context, "detail_proof", scene_goal=first_point)
    result_point = selling_points[2] if len(selling_points) > 2 else _fallback_public_caption(product_context, "cta", scene_goal=first_point)
    title = _product_context_title(product_context) or "这款商品"
    return {
        "narrative_arc": "value_hook -> core_selling_point -> usage_result -> cta",
        "full_subtitle_script": f"{title}，{first_point}，{result_point}",
        "voiceover_script": "",
        "beats": [
            {"start_seconds": 0, "end_seconds": 3, "role": "hook", "message": "使用情境", "subtitle": _fallback_public_caption(product_context, "hook", scene_goal=first_point)},
            {"start_seconds": 3, "end_seconds": duration - 5, "role": "feature_demo", "message": "核心卖点", "subtitle": first_point},
            {"start_seconds": max(3, duration - 5), "end_seconds": duration - 2, "role": "detail_proof", "message": "卖点细节", "subtitle": second_point},
            {"start_seconds": max(5, duration - 2), "end_seconds": duration, "role": "cta", "message": "结果收束", "subtitle": result_point},
        ],
        "closing_intent": f"用结果画面强化「{result_point}」。",
        "hook": _fallback_public_caption(product_context, "hook", scene_goal=first_point),
        "body": selling_points[:3] or [first_point],
        "cta": result_point,
        "style_notes": "保守降级结构",
        "target_duration_seconds": duration,
    }


def _fallback_conservative_storyboard(product_context: dict[str, Any], script_plan: dict[str, Any]) -> list[dict[str, Any]]:
    duration = max(8, int(product_context.get("duration_seconds", 15)))
    beats = script_plan.get("beats", [])
    if not beats:
        beats = [
            {"start_seconds": 0, "end_seconds": 3, "role": "hook", "subtitle": ""},
            {"start_seconds": 3, "end_seconds": duration - 2, "role": "feature_demo", "subtitle": ""},
            {"start_seconds": max(5, duration - 2), "end_seconds": duration, "role": "cta", "subtitle": ""},
        ]
    identity_card = product_context.get("product_identity_card", {})
    storyboard = []
    for i, beat in enumerate(beats):
        shot_duration = max(1, int(beat.get("end_seconds", 0)) - int(beat.get("start_seconds", 0)))
        role = beat.get("role", "hook")
        role_key = str(role or "").strip()
        is_product_shot = role_key in ("feature_demo", "detail_proof", "product_showcase")
        subtitle = _safe_user_caption(
            str(beat.get("subtitle", "")).strip(),
            fallback=_fallback_public_caption(product_context, role_key, scene_goal=str(beat.get("message", "")).strip()),
            max_chars=DEFAULT_SUBTITLE_MAX_CHARS,
        )
        storyboard.append({
            "shot_index": i + 1,
            "narrative_role": role_key,
            "scene_goal": beat.get("message", ""),
            "initial_state": "",
            "action": "轻微推近" if role_key != "cta" else "定格",
            "final_state": "",
            "duration_seconds": shot_duration,
            "subtitle": subtitle,
            "voiceover": subtitle,
            "camera_motion": "定镜" if role_key == "cta" else "推近",
            "visual_description": beat.get("message", ""),
            "render_strategy": "image_to_video" if is_product_shot else "text_to_video",
            "product_presence": "required" if is_product_shot else "optional",
            "identity_strictness": "high" if is_product_shot else "low",
            "asset_requirement": "",
            "review_focus": [],
            "completion_criteria": [],
            "product_identity_constraints": identity_card.get("must_preserve", []),
        })
    return storyboard


def _ensure_storyboard_fields(storyboard: list[dict[str, Any]], product_context: dict[str, Any]) -> list[dict[str, Any]]:
    """确保每个分镜包含审核所需的所有字段，优先使用 image_to_video 而非 text_to_video。"""

    identity_card = product_context.get("product_identity_card", {})
    ref_asset_ids = identity_card.get("reference_asset_ids", [])

    for shot in storyboard:
        is_product_shot = str(shot.get("product_presence", "")).strip().lower() == "required"
        is_identity_high = str(shot.get("identity_strictness", "")).strip().lower() in ("high", "strict")
        # 商品镜头或高身份约束的镜头默认走图生视频
        default_strategy = "image_to_video" if (is_product_shot or is_identity_high) else "text_to_video"
        shot.setdefault("render_strategy", default_strategy)
        shot.setdefault("product_presence", "optional")
        shot.setdefault("identity_strictness", "relaxed")
        shot.setdefault("asset_requirement", "")
        shot.setdefault("review_focus", "")
        shot.setdefault("completion_criteria", "")
        shot.setdefault("action", "轻微推近")
        shot.setdefault("initial_state", "")
        shot.setdefault("final_state", "")
        shot.setdefault("asset_id", "")
        shot.setdefault("asset_usage", {
            "selected_asset_ids": ref_asset_ids if is_product_shot else [],
            "is_identity_critical": is_product_shot,
        })
    return _ensure_storyboard_continuity(storyboard)


def _ensure_storyboard_continuity(storyboard: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """补全安全的连续性配置；不满足同组条件时回退硬切，避免错误复用上一镜尾帧。"""

    result = []
    previous_shot: dict[str, Any] | None = None
    for raw_shot in storyboard:
        shot = dict(raw_shot)
        transition_type = str(shot.get("transition_type", "hard_cut")).strip().lower()
        if transition_type not in {"hard_cut", "crossfade", "continue_from_previous"}:
            transition_type = "hard_cut"

        continuity_group = str(shot.get("continuity_group", "")).strip()
        previous_group = str((previous_shot or {}).get("continuity_group", "")).strip()
        if (
            transition_type == "continue_from_previous"
            and (not previous_shot or not continuity_group or continuity_group != previous_group)
        ):
            transition_type = "hard_cut"

        shot["continuity_group"] = continuity_group
        shot["transition_type"] = transition_type
        if (
            transition_type == "continue_from_previous"
            and str(shot.get("product_presence", "")).strip().lower() == "required"
            and str(shot.get("asset_id", "")).strip()
        ):
            # 商品连续镜在结尾回到真实素材锚点，降低多段续写后的外观累计漂移。
            shot["anchor_last_frame"] = True
        else:
            shot["anchor_last_frame"] = False

        result.append(shot)
        previous_shot = shot
    return result


def _repair_script_by_rules(
    script_plan: dict[str, Any],
    review: dict[str, Any],
    product_context: dict[str, Any],
) -> dict[str, Any]:
    """基于审核发现的问题，用规则修复剧本而非重新调用 LLM。"""
    issues = review.get("issues", [])
    if not issues:
        return script_plan

    script = dict(script_plan)
    duration = int(product_context.get("duration_seconds", 15))

    # 修复 hook 缺失
    for issue in issues:
        issue_str = str(issue).lower()
        if "hook" in issue_str or "开场" in issue_str:
            if not script.get("hook"):
                script["hook"] = _fallback_public_caption(product_context, "hook", scene_goal="使用场景看清")
            beats = script.get("beats", [])
            if beats and not any(b.get("role") == "hook" for b in beats):
                beats.insert(0, {"start_seconds": 0, "end_seconds": 2, "role": "hook", "message": "使用情境", "subtitle": script["hook"]})

    # 修复 CTA 缺失
    for issue in issues:
        issue_str = str(issue).lower()
        if "cta" in issue_str or "结尾" in issue_str:
            if not script.get("cta"):
                script["cta"] = _fallback_public_caption(product_context, "cta", scene_goal="卖点结果看得见")
            beats = script.get("beats", [])
            if beats and not any(b.get("role") == "cta" for b in beats):
                beats.append({"start_seconds": duration - 2, "end_seconds": duration, "role": "cta", "message": "结果收束", "subtitle": script["cta"]})

    # 修复叙事弧线
    if not script.get("narrative_arc"):
        roles = [b.get("role", "") for b in script.get("beats", [])]
        script["narrative_arc"] = " -> ".join(roles) if roles else "hook -> feature_demo -> cta"

    # 修复时长覆盖
    beats = script.get("beats", [])
    if beats:
        beats[-1]["end_seconds"] = min(beats[-1].get("end_seconds", duration), duration)

    return script


def _repair_storyboard_by_rules(
    storyboard: list[dict[str, Any]],
    review: dict[str, Any],
    script_plan: dict[str, Any],
    product_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """基于审核发现的问题，用规则修复分镜而非重新调用 LLM。"""
    issues = review.get("issues", [])
    if not issues:
        return storyboard

    # 确保字段完整
    result = _ensure_storyboard_fields(list(storyboard), product_context)

    # 修复时长不匹配
    for issue in issues:
        issue_str = str(issue).lower()
        if "时长" in issue_str or "duration" in issue_str or "秒" in issue_str:
            total = sum(int(s.get("duration_seconds", 0)) for s in result)
            target = int(product_context.get("duration_seconds", 15))
            if total > 0 and abs(total - target) > 3:
                scale = target / total
                for s in result:
                    s["duration_seconds"] = max(1, round(s.get("duration_seconds", 1) * scale))

    return result

def _normalize_script_plan(
    raw_script: Any,
    product_context: dict[str, Any],
) -> dict[str, Any] | None:
    """把 LLM 返回的剧本 JSON 归一化成页面和分镜节点需要的结构。"""

    if not isinstance(raw_script, dict):
        return None

    identity_card = _safe_dict(product_context.get("product_identity_card", {}))
    expected_product_type = (
        str(identity_card.get("product_type", "")).strip()
        or str(product_context.get("product_title", "")).strip()
    )

    body = raw_script.get("body", [])
    if isinstance(body, str):
        body = [body]
    if not isinstance(body, list):
        body = []

    raw_beats = raw_script.get("beats", [])
    if not isinstance(raw_beats, list):
        raw_beats = []
    raw_context = raw_script.get("context_reconstruction", {})
    if not isinstance(raw_context, dict):
        raw_context = {}
    beats = []
    for beat in raw_beats:
        if not isinstance(beat, dict):
            continue
        try:
            start_s = int(beat.get("start_seconds", 0))
        except (TypeError, ValueError):
            start_s = 0
        try:
            end_s = int(beat.get("end_seconds", 0))
        except (TypeError, ValueError):
            end_s = 0
        beats.append({
            "start_seconds": start_s,
            "end_seconds": end_s,
            "role": str(beat.get("role", "")).strip(),
            "message": str(beat.get("message", "")).strip(),
            "subtitle": str(beat.get("subtitle", "")).strip(),
            "visual_intent": str(beat.get("visual_intent", "")).strip(),
            "evidence_refs": _string_list(beat.get("evidence_refs", [])),
            "scene_before": str(beat.get("scene_before", "")).strip(),
            "action": str(beat.get("action", "")).strip(),
            "scene_after": str(beat.get("scene_after", "")).strip(),
            "visible_entities": _string_list(beat.get("visible_entities", [])),
            "physical_constraints": _string_list(beat.get("physical_constraints", [])),
            "asset_requirements": _string_list(beat.get("asset_requirements", [])),
            "continuity_mode": str(beat.get("continuity_mode", "")).strip(),
            "transition_reason": str(beat.get("transition_reason", "")).strip(),
            "shot_type": str(beat.get("shot_type", "")).strip(),
            "camera_movement": str(beat.get("camera_movement", "")).strip(),
            "scene_description": str(beat.get("scene_description", "")).strip(),
            "subject_appearance": str(beat.get("subject_appearance", "")).strip(),
            "subject_position": str(beat.get("subject_position", "")).strip(),
            "acting_direction": str(beat.get("acting_direction", "")).strip(),
            "dialogue": str(beat.get("dialogue", "")).strip(),
            "scene_elements": _string_list(beat.get("scene_elements", [])),
            "cut_reason": str(beat.get("cut_reason", "")).strip(),
        })

    script = {
        "expected_product_type": expected_product_type,
        "grounded_product_type": str(raw_script.get("grounded_product_type", "")).strip(),
        "product_grounding_summary": str(raw_script.get("product_grounding_summary", "")).strip(),
        "narrative_arc": str(raw_script.get("narrative_arc", "")).strip(),
        "context_reconstruction": {
            "scene_setting": str(raw_context.get("scene_setting", "")).strip(),
            "plot_development": str(raw_context.get("plot_development", "")).strip(),
            "emotional_tendency": str(raw_context.get("emotional_tendency", "")).strip(),
            "speaking_intent": str(raw_context.get("speaking_intent", "")).strip(),
            "causal_chain": _string_list(raw_context.get("causal_chain", [])),
        },
        "story_title": str(raw_script.get("story_title", "")).strip(),
        "rich_story_text": str(raw_script.get("rich_story_text", "")).strip(),
        "core_message": str(raw_script.get("core_message", "")).strip(),
        "user_emotion": str(raw_script.get("user_emotion", "")).strip(),
        "key_visual_moments": _string_list(raw_script.get("key_visual_moments", [])),
        "full_subtitle_script": str(raw_script.get("full_subtitle_script") or raw_script.get("subtitle_script", "")).strip(),
        "subtitle_script": str(raw_script.get("subtitle_script") or raw_script.get("full_subtitle_script", "")).strip(),
        "voiceover_script": str(raw_script.get("voiceover_script", "")).strip(),
        "beats": beats,
        "closing_intent": str(raw_script.get("closing_intent", "")).strip(),
        "hook": str(raw_script.get("hook", "")).strip(),
        "body": [str(line).strip() for line in body if str(line).strip()],
        "cta": str(raw_script.get("cta", "")).strip(),
        "tone": str(raw_script.get("tone", "")).strip() or _merge_style_text(product_context),
        "style_notes": str(raw_script.get("style_notes", "")).strip() or _merge_style_text(product_context),
        "target_duration_seconds": int(raw_script.get("target_duration_seconds", 0) or product_context.get("duration_seconds", 15)),
        "script_contract_version": str(
            raw_script.get("script_contract_version", "v3_paper_style_cinematic_script")
        ).strip(),
    }
    if not script["rich_story_text"]:
        script["rich_story_text"] = script["full_subtitle_script"] or " ".join(
            str(beat.get("message", "")).strip() for beat in beats if str(beat.get("message", "")).strip()
        )
    if not script["core_message"]:
        selling_points = product_context.get("selling_points", [])
        script["core_message"] = script["hook"] or (str(selling_points[0]) if selling_points else "")
    if not script["key_visual_moments"]:
        script["key_visual_moments"] = [
            str(beat.get("visual_intent") or beat.get("message", "")).strip()
            for beat in beats
            if str(beat.get("visual_intent") or beat.get("message", "")).strip()
        ]
    if not any([script["hook"], script["body"], script["cta"], script["rich_story_text"], script["beats"]]):
        return None
    return script


def _string_list(value: Any) -> list[str]:
    """把模型可能返回的字符串/数组统一成字符串数组。"""

    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _safe_dict(value: Any) -> dict[str, Any]:
    """外部模型偶尔会把对象返回成数组；这里统一兜底为空字典，避免 `.get()` 崩溃。"""

    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float) -> float:
    """把外部输入安全转成浮点数，失败时返回默认值。"""

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_asset_usage(raw_usage: Any) -> dict[str, Any]:
    """归一化分镜素材使用方式，保证后续匹配和页面展示都有稳定字段。"""

    if not isinstance(raw_usage, dict):
        raw_usage = {}
    return {
        "usage_type": str(raw_usage.get("usage_type", "")).strip(),
        "required_asset_role": str(raw_usage.get("required_asset_role", "")).strip(),
        "selected_asset_ids": _string_list(raw_usage.get("selected_asset_ids", [])),
        "is_identity_critical": bool(raw_usage.get("is_identity_critical", False)),
        "can_generate_without_asset": bool(raw_usage.get("can_generate_without_asset", True)),
        "reason": str(raw_usage.get("reason", "")).strip(),
    }


def _infer_default_product_presence(narrative_role: str) -> str:
    role_map = {
        "hook": "optional",
        "feature_demo": "required",
        "detail_proof": "required",
        "cta": "required",
    }
    return role_map.get(narrative_role, "optional")


def _infer_default_identity_strictness(product_presence: str) -> str:
    strictness_map = {
        "required": "high",
        "optional": "low",
        "forbidden": "low",
    }
    return strictness_map.get(product_presence, "medium")


def _normalize_storyboard(raw_storyboard: Any) -> list[dict[str, Any]]:
    """把 LLM 返回的分镜 JSON 归一化成统一列表。"""

    if isinstance(raw_storyboard, dict):
        raw_storyboard = raw_storyboard.get("storyboard") or raw_storyboard.get("shots")
    if not isinstance(raw_storyboard, list):
        return []

    normalized: list[dict[str, Any]] = []
    for index, raw_shot in enumerate(raw_storyboard, start=1):
        if not isinstance(raw_shot, dict):
            continue

        try:
            duration_seconds = int(raw_shot.get("duration_seconds", 0))
        except (TypeError, ValueError):
            duration_seconds = 0

        asset_usage = _normalize_asset_usage(raw_shot.get("asset_usage", {}))
        raw_asset_id = str(raw_shot.get("asset_id", "")).strip()
        if not raw_asset_id and asset_usage.get("selected_asset_ids"):
            raw_asset_id = asset_usage["selected_asset_ids"][0]
        if raw_asset_id and raw_asset_id not in asset_usage["selected_asset_ids"]:
            asset_usage["selected_asset_ids"].insert(0, raw_asset_id)
        narrative_role = _normalize_narrative_role(raw_shot.get("narrative_role", ""))
        subtitle, voiceover = _normalize_storyboard_caption(raw_shot, narrative_role)
        normalized.append(
            {
                "shot_index": int(raw_shot.get("shot_index") or index),
                "duration_seconds": duration_seconds,
                "purpose": str(raw_shot.get("purpose", "")).strip(),
                "narrative_role": narrative_role,
                "continuity_mode": str(raw_shot.get("continuity_mode", "")).strip(),
                "continuity_group": str(raw_shot.get("continuity_group") or "").strip(),
                "transition_type": str(raw_shot.get("transition_type") or "hard_cut").strip() or "hard_cut",
                "anchor_last_frame": bool(raw_shot.get("anchor_last_frame")),
                "scene_goal": str(raw_shot.get("scene_goal", "")).strip(),
                "initial_state": str(raw_shot.get("initial_state", "")).strip(),
                "action": str(raw_shot.get("action", "")).strip(),
                "final_state": str(raw_shot.get("final_state", "")).strip(),
                "camera_motion": str(raw_shot.get("camera_motion", "")).strip(),
                "visual_description": str(raw_shot.get("visual_description", "")).strip()
                or " ".join(
                    v
                    for v in [
                        str(raw_shot.get("scene_goal", "")).strip(),
                        str(raw_shot.get("action", "")).strip(),
                        str(raw_shot.get("initial_state", "")).strip(),
                        str(raw_shot.get("final_state", "")).strip(),
                    ]
                    if v
                ).strip(),
                "subtitle": subtitle,
                "voiceover": voiceover,
                "asset_id": raw_asset_id,
                "asset_requirement": str(raw_shot.get("asset_requirement", "")).strip(),
                "product_identity_constraints": _string_list(raw_shot.get("product_identity_constraints", [])),
                "asset_usage": asset_usage,
                "generation_mode": str(raw_shot.get("generation_mode", "")).strip(),
                "video_prompt": str(raw_shot.get("video_prompt", "")).strip(),
                "force_video_prompt": bool(raw_shot.get("force_video_prompt")),
                "final_prompt_source": str(raw_shot.get("final_prompt_source", "")).strip(),
                "selected_prompt_skill": str(raw_shot.get("selected_prompt_skill", "")).strip(),
                "planner_source": str(raw_shot.get("planner_source", "")).strip(),
                "material_strategy": str(raw_shot.get("material_strategy", "")).strip(),
                "plan_contract": _safe_dict(raw_shot.get("plan_contract", {})),
                "risk_notes": _string_list(raw_shot.get("risk_notes", [])),
                "render_strategy": str(raw_shot.get("render_strategy", "image_to_video")).strip(),
                "product_presence": str(raw_shot.get("product_presence", "")).strip().lower(),
                "identity_strictness": str(raw_shot.get("identity_strictness", "")).strip().lower(),
                "allowed_variation": _string_list(raw_shot.get("allowed_variation", [])),
                "forbidden_variation": _string_list(raw_shot.get("forbidden_variation", [])),
                "review_focus": _string_list(raw_shot.get("review_focus", [])),
                "completion_criteria": _string_list(raw_shot.get("completion_criteria", [])),
            }
        )

    product_identity_card = {}
    for shot in normalized:
        product_identity_constraints = shot.get("product_identity_constraints", [])
        if product_identity_constraints and not product_identity_card:
            product_identity_card = {"must_preserve": product_identity_constraints}

    for shot in normalized:
        narrative_role = str(shot.get("narrative_role", "")).strip()
        raw_product_presence = str(shot.get("product_presence", "")).strip().lower()
        if raw_product_presence in ("required", "optional", "forbidden"):
            product_presence = raw_product_presence
        else:
            product_presence = _infer_default_product_presence(narrative_role)
        shot["product_presence"] = product_presence

        raw_identity_strictness = str(shot.get("identity_strictness", "")).strip().lower()
        if raw_identity_strictness in ("high", "medium", "low"):
            identity_strictness = raw_identity_strictness
        else:
            identity_strictness = _infer_default_identity_strictness(product_presence)
        shot["identity_strictness"] = identity_strictness

        allowed = shot.get("allowed_variation", [])
        if not isinstance(allowed, list):
            allowed = [str(allowed)] if str(allowed).strip() else []
        shot["allowed_variation"] = [str(item).strip() for item in allowed if str(item).strip()]

        forbidden = shot.get("forbidden_variation", [])
        if not isinstance(forbidden, list):
            forbidden = [str(forbidden)] if str(forbidden).strip() else []
        forbidden = [str(item).strip() for item in forbidden if str(item).strip()]
        must_preserve = product_identity_card.get("must_preserve", [])
        for item in must_preserve:
            item_str = str(item).strip()
            if item_str and item_str not in forbidden:
                forbidden.append(item_str)
        shot["forbidden_variation"] = forbidden

        review = shot.get("review_focus", [])
        if not isinstance(review, list):
            review = [str(review)] if str(review).strip() else []
        shot["review_focus"] = [str(item).strip() for item in review if str(item).strip()]

        criteria = shot.get("completion_criteria", [])
        if not isinstance(criteria, list):
            criteria = [str(criteria)] if str(criteria).strip() else []
        shot["completion_criteria"] = [str(item).strip() for item in criteria if str(item).strip()]

    return normalized


def _normalize_storyboard_caption(raw_shot: dict[str, Any], narrative_role: str) -> tuple[str, str]:
    """清理字幕和口播，保留完整用户可见短句，避免烧录内部镜头说明。"""

    fallback_by_role = {
        "hook": "使用场景看清",
        "problem": "使用场景看清",
        "context": "场景自然承接",
        "product_reveal": "轻巧登场",
        "feature_demo": "核心卖点，一眼看懂",
        "detail_proof": "细节经得起近看",
        "lifestyle_result": "体验更进一步",
        "cta": "卖点结果看得见",
    }
    fallback = fallback_by_role.get(narrative_role, "重点一眼看懂")
    raw_subtitle = str(raw_shot.get("subtitle") or "").strip()
    raw_voiceover = str(raw_shot.get("voiceover") or "").strip()
    raw_scene_goal = str(raw_shot.get("scene_goal") or "").strip()
    subtitle_source = raw_subtitle or raw_voiceover or raw_scene_goal
    subtitle = _safe_user_caption(
        subtitle_source,
        fallback=fallback,
        max_chars=DEFAULT_SUBTITLE_MAX_CHARS,
    )
    voiceover_source = raw_voiceover or raw_subtitle or raw_scene_goal
    voiceover = _clean_short_sentence(voiceover_source, max_chars=DEFAULT_VOICEOVER_MAX_CHARS)
    if not voiceover or _is_internal_or_generic_caption(voiceover):
        voiceover = subtitle
    return subtitle, voiceover


def _normalize_narrative_role(raw_role: Any) -> str:
    """归一化导演模型常见角色别名，避免同义词破坏后续规则判断。"""

    role = str(raw_role or "").strip().lower()
    aliases = {
        "proof": "detail_proof",
        "detail": "detail_proof",
        "feature": "feature_demo",
        "demo": "feature_demo",
        "reveal": "product_reveal",
        "closing": "cta",
    }
    return aliases.get(role, role)


def _detect_asset_type(content_type: str) -> str:
    """根据 Content-Type 判断素材类型。"""

    if content_type.startswith("image/"):
        return "image"
    if content_type.startswith("video/"):
        return "video"
    return "unknown"


def _suggest_asset_role(asset_type: str) -> str:
    """基于素材类型给出确定性的基础角色，语义用途仍交给 LLM。"""

    if asset_type == "image":
        return "商品图或细节图候选"
    if asset_type == "video":
        return "商品视频片段候选"
    return "暂不支持的素材"


def _shot_prefers_real_asset(shot: dict[str, Any]) -> bool:
    """判断一个分镜是否应该优先绑定真实上传素材。"""

    asset_usage = shot.get("asset_usage") or {}
    if asset_usage.get("is_identity_critical") is True:
        return True

    product_presence = str(shot.get("product_presence", "")).strip().lower()
    identity_strictness = str(shot.get("identity_strictness", "")).strip().lower()
    # 商品必须出现或身份一致性要求高时，优先绑定真实上传素材，不能被 LLM 的 text_to_video 策略覆盖。
    if product_presence == "required" or identity_strictness == "high":
        return True

    llm_strategy = str(shot.get("render_strategy", "")).strip()
    if llm_strategy in {"text_to_video", "ai_image_then_video"}:
        return False

    text = (
        str(shot.get("purpose", ""))
        + str(shot.get("visual_description", ""))
        + str(shot.get("asset_requirement", ""))
    ).lower()
    real_asset_keywords = [
        "商品",
        "产品",
        "外观",
        "细节",
        "材质",
        "logo",
        "商标",
        "品牌标识",
        "包装",
        "真实",
        "主图",
        "近景",
        "特写",
    ]
    scene_keywords = ["场景", "氛围", "生活方式", "lifestyle", "背景", "转场", "情绪"]
    return any(keyword in text for keyword in real_asset_keywords) and not any(
        keyword in text for keyword in scene_keywords
    )


def _build_gap_completion_prompt(shot: dict[str, Any]) -> str:
    """生成给文生视频/后续文生图使用的素材补全 prompt。"""

    visual = str(shot.get("visual_description", "")).strip()
    purpose = str(shot.get("purpose", "")).strip()
    requirement = str(shot.get("asset_requirement", "")).strip()
    return (
        f"{visual}\n"
        f"镜头目的：{purpose}\n"
        f"素材要求：{requirement}\n"
        "画面必须真实、干净、无文字、无水印；如果缺少真实商品图，不要编造具体 logo 或品牌文字。"
    )


def _asset_gap_record(
    shot: dict[str, Any],
    original_strategy: str,
    updated_match: dict[str, Any],
) -> dict[str, Any]:
    """记录素材缺口从发现到补全的过程，方便页面和 Trace 展示。"""

    return {
        "shot_index": shot.get("shot_index"),
        "purpose": shot.get("purpose", ""),
        "original_strategy": original_strategy,
        "final_strategy": updated_match.get("strategy", ""),
        "completion_type": updated_match.get("completion_type", ""),
        "status": updated_match.get("match_status", ""),
        "risk": updated_match.get("risk", ""),
        "note": updated_match.get("note", ""),
    }


def _choose_render_strategy(shot: dict[str, Any], matched_asset: dict[str, Any] | None) -> str:
    """为单个分镜选择视频生成策略。优先采用 LLM 的决定，其次用规则兜底。"""

    llm_strategy = str(shot.get("render_strategy", "")).strip()
    valid_strategies = {"image_to_video", "text_to_video", "ai_image_then_video", "crop_and_ken_burns"}

    product_presence = str(shot.get("product_presence", "")).strip().lower()
    identity_strictness = str(shot.get("identity_strictness", "")).strip().lower()
    if product_presence == "forbidden":
        return "text_to_video"
    if matched_asset and matched_asset.get("file_path"):
        if llm_strategy == "crop_and_ken_burns":
            return "crop_and_ken_burns"
        return "image_to_video"

    if (
        str(shot.get("material_strategy", "")).strip() == "ideal_commerce_scene"
        and str(shot.get("planner_source", "")).startswith("B_ideal_commerce_scene")
        and llm_strategy == "text_to_video"
    ):
        return "text_to_video"

    # 商品主镜头没有真实素材时禁止降级为 text_to_video，避免凭空生成错误商品。
    if product_presence == "required" or identity_strictness == "high":
        return "needs_user_asset"

    # LLM 选好了且有对应素材时直接采用
    if llm_strategy == "text_to_video":
        return "text_to_video"
    if llm_strategy == "ai_image_then_video":
        return "ai_image_then_video"
    if llm_strategy == "needs_user_asset":
        return "needs_user_asset"

    # LLM 策略不适用时回退到规则
    if matched_asset and matched_asset.get("file_path"):
        return "image_to_video"

    text = (
        str(shot.get("purpose", ""))
        + str(shot.get("visual_description", ""))
        + str(shot.get("asset_requirement", ""))
    )
    product_keywords = ["商品主体", "产品主体", "外观", "细节", "logo", "商标", "品牌标识", "包装"]
    if any(keyword in text for keyword in product_keywords):
        return "needs_user_asset"
    return "text_to_video"


def _asset_match_note(strategy: str, matched_asset: dict[str, Any] | None) -> str:
    """说明当前分镜为什么选择该渲染策略。"""

    if strategy == "image_to_video":
        filename = matched_asset.get("filename", "上传素材") if matched_asset else "上传素材"
        return f"使用 {filename} 作为图生视频输入。"
    if strategy == "text_to_video":
        return "没有合适上传素材，使用分镜画面描述走文生视频。"
    return "该分镜需要真实商品素材，当前素材不足，建议用户补充图片。"


def _fallback_asset_summary(
    assets: list[dict[str, Any]],
    task_data: dict[str, Any],
) -> str:
    """当 LLM 不可用时，给页面一个明确的降级说明。"""

    if not assets:
        return "未上传素材，后续分镜只能先生成文案和画面需求。"

    supported_count = sum(1 for asset in assets if asset["is_supported"])
    if supported_count:
        return (
            f"已收到 {supported_count} 个图片或视频素材。"
            "当前未启用多模态模型，暂只能基于文件类型和上传顺序判断用途，不使用文件名推断画面内容。"
        )
    return "已收到文件，但没有识别到支持的图片或视频素材。"


def _call_text_llm(prompt_data: dict[str, Any], purpose: str, temperature: float = 0.7) -> dict[str, Any]:
    """
    调用文本模型，按优先级依次尝试多个后端，失败时自动 fallback。

    高影响创意节点优先使用火山方舟 Ark；低风险结构化节点优先使用 DeepSeek。
    每个后端可独立配置，未配置则跳过。所有后端都不可用或全部失败才返回失败。
    """

    if os.getenv("AIGC_DISABLE_LLM") == "1":
        return {
            "ok": False,
            "content": "",
            "error": "当前通过 AIGC_DISABLE_LLM=1 禁用了模型调用。",
        }

    timeout = int(os.getenv("TEXT_LLM_TIMEOUT", "60"))

    # 先收集可用后端，再根据任务风险排序。
    backends: list[tuple[str, str, str, str]] = []
    deepseek_backend: tuple[str, str, str, str] | None = None
    ark_backend: tuple[str, str, str, str] | None = None

    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if deepseek_key:
        deepseek_backend = (
            "deepseek",
            os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
            deepseek_key,
            os.getenv("DEEPSEEK_MODEL", "deepseek-chat").strip(),
        )

    ark_key = os.getenv("ARK_API_KEY", "").strip()
    ark_endpoint_id = os.getenv("ARK_TEXT_ENDPOINT_ID", "").strip()
    if ark_key and ark_endpoint_id:
        ark_backend = (
            "ark",
            "https://ark.cn-beijing.volces.com/api/v3",
            ark_key,
            ark_endpoint_id,
        )

    creative_purposes = {"script_plan", "director_storyboard"}
    if purpose.split("[", 1)[0] in creative_purposes:
        # 剧本和导演分镜会直接影响视频内容，优先交给官方多模态模型对应的方舟端点。
        backends.extend(backend for backend in (ark_backend, deepseek_backend) if backend)
    else:
        backends.extend(backend for backend in (deepseek_backend, ark_backend) if backend)

    # 本地 OpenAI 兼容端点（Ollama / vLLM，兜底）
    local_url = os.getenv("TEXT_LLM_BASE_URL", "").strip()
    local_model = os.getenv("TEXT_LLM_MODEL", "").strip()
    local_key = os.getenv("TEXT_LLM_API_KEY", "not-needed").strip()
    if local_url and local_model:
        backends.append(("local", local_url, local_key, local_model))

    if not backends:
        return {
            "ok": False,
            "content": "",
            "error": "未配置任何文本 LLM 后端。请设置 TEXT_LLM_BASE_URL+TEXT_LLM_MODEL 或 DEEPSEEK_API_KEY 或 ARK_API_KEY+ARK_TEXT_ENDPOINT_ID。",
        }

    errors: list[str] = []
    for backend_name, base_url, api_key, model in backends:
        print(f"[video_generation_workflow] 尝试 {backend_name} 后端：model={model}, purpose={purpose}", flush=True)
        result = _call_openai_compatible_api(
            base_url=base_url,
            api_key=api_key,
            model=model,
            prompt_data=prompt_data,
            purpose=f"{purpose}[{backend_name}]",
            temperature=temperature,
            timeout=timeout,
        )
        if result["ok"]:
            print(f"[video_generation_workflow] {backend_name} 调用成功", flush=True)
            return result
        errors.append(f"{backend_name}: {result.get('error', 'unknown')}")
        print(f"[video_generation_workflow] {backend_name} 失败，尝试下一个后端。", flush=True)

    return {
        "ok": False,
        "content": "",
        "error": "所有文本 LLM 后端均调用失败：" + "；".join(errors),
    }


def _call_multimodal_llm(
    prompt_data: dict[str, Any],
    image_paths: list[str],
    purpose: str,
) -> dict[str, Any]:
    """调用火山方舟多模态模型，让模型真实读取上传图片。"""

    if os.getenv("AIGC_DISABLE_LLM") == "1":
        return {
            "ok": False,
            "content": "",
            "error": "当前通过 AIGC_DISABLE_LLM=1 禁用了模型调用。",
        }

    ark_key = os.getenv("ARK_API_KEY")
    ark_endpoint_id = os.getenv("ARK_TEXT_ENDPOINT_ID")
    if not ark_key or not ark_endpoint_id:
        return {
            "ok": False,
            "content": "",
            "error": "未配置 ARK_API_KEY 或 ARK_TEXT_ENDPOINT_ID，无法进行多模态素材理解。",
        }

    image_batches = _build_multimodal_image_batches(image_paths)
    if not image_batches:
        return {
            "ok": False,
            "content": "",
            "error": "没有可读取的图片文件，无法进行多模态素材理解。",
        }

    prompt_with_guardrails = dict(prompt_data)
    prompt_with_guardrails.setdefault(
        "product_type_disambiguation_rule",
        (
            "必须区分容易混淆的商品类型。例如“笔记本”必须判断是纸质笔记本 notebook "
            "还是笔记本电脑 laptop，并依据图片中可见的屏幕、键盘、铰链、Logo 和结构判断。"
        ),
    )

    last_result: dict[str, Any] | None = None
    for attempt_index, image_data_urls in enumerate(image_batches, start=1):
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(prompt_with_guardrails, ensure_ascii=False),
            }
        ]
        for data_url in image_data_urls:
            content.append({"type": "image_url", "image_url": {"url": data_url}})

        payload = {
            "model": ark_endpoint_id,
            "temperature": 0.2,
            # 方舟 Chat API 支持 JSON Object 模式。素材理解依赖结构化字段，
            # 这里必须要求模型返回合法 JSON，避免视觉事实因解析失败被整体丢弃。
            "response_format": {"type": "json_object"},
            "thinking": {"type": "disabled"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是电商 AIGC 带货视频系统的素材分析模型。"
                        "请根据图片内容判断商品主体、画面质量、可用于哪些分镜、有什么风险。"
                        "输出合法 JSON，具体、克制、不要编造看不见的信息。"
                    ),
                },
                {
                    "role": "user",
                    "content": content,
                },
            ],
        }
        last_result = _post_openai_compatible_payload(
            base_url="https://ark.cn-beijing.volces.com/api/v3",
            api_key=ark_key,
            payload=payload,
            purpose=purpose,
            timeout=120,
        )
        if last_result["ok"]:
            last_result["multimodal_attempt"] = attempt_index
            return last_result

    return {
        "ok": False,
        "content": "",
        "error": f"多模态模型调用失败，已尝试原图和压缩图：{(last_result or {}).get('error')}",
    }


def _build_multimodal_image_batches(image_paths: list[str]) -> list[list[str]]:
    """构造多模态请求的图片批次：先原图，失败后压缩 JPEG。"""

    original_urls = []
    compressed_urls = []
    for image_path in image_paths:
        original_url = _image_file_to_data_url(image_path)
        if original_url:
            original_urls.append(original_url)
        compressed_url = _compressed_image_to_data_url(image_path)
        if compressed_url:
            compressed_urls.append(compressed_url)

    batches = []
    if original_urls:
        batches.append(original_urls)
    if compressed_urls and compressed_urls != original_urls:
        batches.append(compressed_urls)
    return batches


def _compressed_image_to_data_url(image_path: str, max_side: int = 1024, quality: int = 86) -> str:
    """把图片压缩成模型更稳定接受的 JPEG data URL。"""

    try:
        from PIL import Image
    except ImportError:
        return ""

    try:
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((max_side, max_side), Image.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=quality, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return ""


def _call_openai_compatible_api(
    base_url: str,
    api_key: str,
    model: str,
    prompt_data: dict[str, Any],
    purpose: str,
    temperature: float = 0.7,
    timeout: int = 60,
) -> dict[str, Any]:
    """调用 OpenAI Chat Completions 兼容接口。"""

    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        # 下游节点依赖结构化字段，统一要求文本模型返回合法 JSON。
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是电商带货视频导演 Agent，拥有 5 年 TikTok Shop 短视频实战经验。"
                    "你精通：3C 数码、美妆个护、家居日用、服饰配饰四大类目的爆款公式。"
                    "你的工作方式：先分析商品定位和目标人群，再决定创意策略，最后生成可执行的分镜方案。"
                    "输出要求：中文、具体、克制、可执行。每一个画面描述都要能用镜头语言实现。"
                    "运动描述使用精确的物理语言——不写「旋转展示」，写「以产品为轴心缓慢水平旋转约 15 度」。"
                    "不编造看不见的信息，不夸大商品效果。"
                    "每条视频应该有独特的创意切入角度，避免反复使用相同的套路和句式。"
                    "最终只返回合法 JSON，不要输出解释文字。"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(prompt_data, ensure_ascii=False),
            },
        ],
    }
    return _post_openai_compatible_payload(
        base_url=base_url,
        api_key=api_key,
        payload=payload,
        purpose=purpose,
        timeout=timeout,
    )


def _post_openai_compatible_payload(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    purpose: str,
    timeout: int = 60,
) -> dict[str, Any]:
    """发送 OpenAI 兼容格式请求并解析响应。"""

    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    print(
        "[video_generation_workflow] 调用模型："
        f"purpose={purpose}, model={payload.get('model')}, timeout={timeout}s",
        flush=True,
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return {
            "ok": False,
            "content": "",
            "error": f"模型接口 HTTP 错误：status={exc.code}",
        }
    except URLError as exc:
        return {
            "ok": False,
            "content": "",
            "error": f"模型接口网络错误：{exc.reason}",
        }
    except TimeoutError:
        return {
            "ok": False,
            "content": "",
            "error": "模型接口请求超时。",
        }
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "content": "",
            "error": f"模型响应解析失败：{exc}",
        }

    content = response_data["choices"][0]["message"]["content"]
    return {
        "ok": True,
        "content": content,
        "error": None,
    }


def _step(name: str, status: str, message: str) -> dict[str, str]:
    """生成统一的工作流步骤记录。"""

    return {
        "name": name,
        "status": status,
        "message": message,
    }


def _print_stage_elapsed(task_id: str, stage_name: str, started_at: float) -> None:
    """输出单个工作流阶段的执行耗时，便于在控制台定位慢步骤。"""

    _flow_print(
        f"[video_generation_workflow] {stage_name}完成："
        f"task_id={task_id}, elapsed={_elapsed_seconds(started_at)}s"
    )


def _elapsed_seconds(started_at: float) -> str:
    """把 perf_counter 的差值格式化成固定两位小数。"""

    return f"{time.perf_counter() - started_at:.2f}"


def _image_file_to_data_url(image_path: str) -> str:
    """把本地图片转换成多模态接口可用的 data URL。"""

    path = Path(image_path)
    if not path.exists():
        return ""

    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _task_output_dir(task_data: dict[str, Any], task_id: str) -> str:
    """根据已保存素材推断当前任务的输出目录。"""

    for asset in task_data.get("uploaded_assets", []):
        file_path = asset.get("file_path")
        if file_path:
            return str(Path(file_path).parent)
    return str(Path(".uploads") / task_id)


def _save_workflow_artifacts(
    task_id: str,
    output_dir: str,
    artifacts: dict[str, Any],
) -> str:
    """把工作流中间产物持久化到磁盘，方便后期回溯排查。

    保存内容：
    - 每个关键阶段的 JSON 产物（格式化的，方便人工阅读和 diff）
    - 原始 LLM 响应文本（从 llm_notes 字段抽取）
    - 工作流执行元信息（时间戳、步骤状态等）

    返回 artifacts 目录路径。
    """

    artifacts_dir = Path(output_dir) / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    # ---- 阶段产物（编号排序，便于按执行顺序阅读） ----
    stage_artifacts: list[tuple[str, Any]] = [
        ("01_asset_analysis", artifacts.get("asset_analysis")),
        ("02_structured_requirements", artifacts.get("structured_requirements")),
        ("03_product_identity_card", artifacts.get("product_identity_card")),
        ("04_product_context", artifacts.get("product_context")),
        ("05_director_decision", artifacts.get("director_decision")),
        ("06_script_plan", artifacts.get("script_plan")),
        ("07_script_review", artifacts.get("script_review")),
        ("08_storyboard", artifacts.get("storyboard")),
        ("09_storyboard_review", artifacts.get("storyboard_review")),
        ("10_narrative_review", artifacts.get("narrative_review")),
        ("10b_narrative_review_attempts", artifacts.get("narrative_review_attempts")),
        ("11_asset_matching", artifacts.get("asset_matching")),
        ("12_asset_gap_completion", artifacts.get("asset_gap_completion")),
        ("13_creation_plan", artifacts.get("creation_plan")),
        ("14_render_result", artifacts.get("render_result")),
        ("15_content_review", artifacts.get("content_review")),
        ("16_final_check", artifacts.get("final_check")),
        ("17_ab_variants", artifacts.get("ab_variants")),
    ]

    for filename, data in stage_artifacts:
        if data is None:
            continue
        path = artifacts_dir / f"{timestamp}_{filename}.json"
        _write_json_artifact(path, data)

    # ---- 原始 LLM 响应（方便查看模型实际输出了什么） ----
    llm_dir = artifacts_dir / "llm_raw_responses"
    llm_dir.mkdir(parents=True, exist_ok=True)

    llm_sources: list[tuple[str, Any]] = [
        ("asset_analysis", artifacts.get("asset_analysis")),
        ("script_plan", artifacts.get("script_plan")),
        ("storyboard", artifacts.get("storyboard")),
        ("director_decision", artifacts.get("director_decision")),
        ("narrative_review", artifacts.get("narrative_review")),
        ("content_review", artifacts.get("content_review")),
    ]

    def _save_llm_raw(name: str, data: Any) -> None:
        raw_text = ""
        if isinstance(data, dict):
            raw_text = str(data.get("llm_notes", "") or data.get("semantic_summary", ""))
        elif isinstance(data, list):
            parts = []
            for item in data:
                if isinstance(item, dict):
                    visual = str(item.get("visual_description", ""))
                    if visual:
                        parts.append(visual)
            raw_text = "\n---\n".join(parts)
        if raw_text and len(raw_text.strip()) > 10:
            (llm_dir / f"{timestamp}_{name}_raw.txt").write_text(raw_text.strip(), encoding="utf-8")

    for name, data in llm_sources:
        _save_llm_raw(name, data)

    # ---- Seedance 渲染 prompt（每个分镜实际发给视频模型的内容） ----
    seedance_dir = artifacts_dir / "seedance_prompts"
    seedance_dir.mkdir(parents=True, exist_ok=True)
    render_result = artifacts.get("render_result") or {}
    shot_results = render_result.get("shot_results", []) if isinstance(render_result, dict) else []
    if shot_results:
        for i, shot in enumerate(shot_results):
            if not isinstance(shot, dict):
                continue
            prompt_text = shot.get("seedance_prompt", "")
            if prompt_text:
                path = seedance_dir / f"{timestamp}_shot_{i + 1:02d}_prompt.txt"
                path.write_text(prompt_text, encoding="utf-8")

    # ---- 工作流执行元信息 ----
    trace: dict[str, Any] = {
        "task_id": task_id,
        "saved_at": timestamp,
        "workflow_status": artifacts.get("workflow_status", "unknown"),
        "workflow_stage": artifacts.get("workflow_stage", "unknown"),
        "workflow_message": artifacts.get("workflow_message", ""),
        "workflow_steps": artifacts.get("workflow_steps", []),
        "trace_summary": artifacts.get("trace_summary"),
    }
    _write_json_artifact(artifacts_dir / f"{timestamp}_workflow_trace.json", trace)

    _flow_print(f"[video_generation_workflow] 工作流中间产物已保存到 {artifacts_dir}")
    return str(artifacts_dir)


def _write_json_artifact(path: Path, data: Any) -> None:
    """写入格式化 JSON 文件，确保中文可读。"""

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
