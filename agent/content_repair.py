"""Content-review repair loop for rendered video shots."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Callable


SUPPORTED_REPAIR_STRATEGIES = {
    "rerender_with_stronger_identity_anchor",
    "simplify_action",
    "rewrite_shot_goal",
    "fallback_to_local_identity_anchor",
}


def repair_rendered_content(
    task_id: str,
    repair_records: list[dict[str, Any]],
    creation_plan: dict[str, Any],
    render_result: dict[str, Any],
    output_dir: str,
    report: Callable[..., None],
    repair_func: Callable[..., dict[str, Any]],
    flow_print: Callable[[str], None],
) -> dict[str, Any]:
    """Execute automatic repair actions and re-concatenate the final video."""

    summary: dict[str, Any] = _empty_repair_summary()
    if not repair_records:
        return summary

    shots = list(creation_plan.get("shots", []))
    if not shots:
        return summary

    output_path = Path(output_dir)
    repaired_any = False

    for record in repair_records:
        shot_index = int(record.get("shot_index", 0))
        repair_strategy = str(record.get("repair_strategy", ""))
        if not repair_strategy:
            _record_skipped(summary, shot_index, repair_strategy, "缺少修复策略")
            continue

        shot = _find_shot(shots, shot_index)
        if shot is None:
            _record_skipped(summary, shot_index, repair_strategy, "没有找到对应分镜")
            continue

        if repair_strategy not in SUPPORTED_REPAIR_STRATEGIES:
            _record_skipped(summary, shot_index, repair_strategy, f"不支持的修复策略：{repair_strategy}")
            continue

        if _anchor_repair_is_inapplicable(repair_strategy, shot):
            flow_print(
                "[video_generation_workflow] 跳过不适用的商品锚点修复："
                f"shot_{shot_index}, product_presence={shot.get('product_presence')}, "
                f"render_strategy={shot.get('render_strategy')}"
            )
            _record_skipped(summary, shot_index, repair_strategy, "该分镜不适用商品锚点重渲染")
            continue

        shot_with_card = _shot_with_product_identity(shot, creation_plan)
        flow_print(f"[video_generation_workflow] 修复 shot_{shot_index}：{repair_strategy}")
        summary["attempted_count"] += 1
        repair_result = repair_func(
            shot=shot_with_card,
            shot_index=shot_index,
            repair_strategy=repair_strategy,
            task_id=task_id,
            output_dir=output_dir,
        )

        if repair_result.get("success") and repair_result.get("clip_path"):
            repaired_any = _copy_repaired_clip(
                summary=summary,
                output_path=output_path,
                shot_index=shot_index,
                repair_strategy=repair_strategy,
                repair_result=repair_result,
                report=report,
                flow_print=flow_print,
            ) or repaired_any
        else:
            error = repair_result.get("error", "未知错误")
            flow_print(f"[video_generation_workflow] shot_{shot_index} 修复失败：{error}")
            _record_failed(summary, shot_index, repair_strategy, error)

    if repaired_any:
        _reconcat_repaired_video(summary, output_path, shots, flow_print)
    return summary


def _empty_repair_summary() -> dict[str, Any]:
    return {
        "attempted_count": 0,
        "succeeded_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "reconcat_success": False,
        "records": [],
    }


def _find_shot(shots: list[dict[str, Any]], shot_index: int) -> dict[str, Any] | None:
    for shot in shots:
        if int(shot.get("shot_index", 0)) == shot_index:
            return shot
    return None


def _anchor_repair_is_inapplicable(repair_strategy: str, shot: dict[str, Any]) -> bool:
    if repair_strategy not in ("rerender_with_stronger_identity_anchor", "fallback_to_local_identity_anchor"):
        return False
    product_presence = str(shot.get("product_presence", "required")).strip().lower()
    render_strategy = str(shot.get("render_strategy", "image_to_video")).strip()
    return product_presence != "required" or render_strategy != "image_to_video"


def _shot_with_product_identity(shot: dict[str, Any], creation_plan: dict[str, Any]) -> dict[str, Any]:
    shot_with_card = dict(shot)
    if shot_with_card.get("product_identity_card"):
        return shot_with_card
    product_identity_card = creation_plan.get("product_identity_card") or product_context_for_repair(creation_plan)
    if product_identity_card:
        shot_with_card["product_identity_card"] = product_identity_card
    return shot_with_card


def product_context_for_repair(creation_plan: dict[str, Any]) -> dict[str, Any]:
    """Recover product identity context from a creation plan."""

    identity_card = creation_plan.get("product_identity_card") or {}
    if identity_card:
        return identity_card
    for shot in creation_plan.get("shots", []):
        identity_card = shot.get("product_identity_card")
        if identity_card:
            return identity_card
    return {}


def _copy_repaired_clip(
    summary: dict[str, Any],
    output_path: Path,
    shot_index: int,
    repair_strategy: str,
    repair_result: dict[str, Any],
    report: Callable[..., None],
    flow_print: Callable[[str], None],
) -> bool:
    old_clip = output_path / f"seedance_shot_{shot_index:02d}.mp4"
    new_clip = Path(repair_result["clip_path"])
    if not new_clip.exists():
        error = f"修复片段不存在：{new_clip}"
        flow_print(f"[video_generation_workflow] shot_{shot_index} {error}")
        _record_failed(summary, shot_index, repair_strategy, error)
        return False

    shutil.copyfile(new_clip, old_clip)
    summary["succeeded_count"] += 1
    summary["records"].append(
        {
            "shot_index": shot_index,
            "status": "succeeded",
            "repair_strategy": repair_strategy,
            "clip_path": str(old_clip),
            "source_clip_path": str(new_clip),
        }
    )
    report("content_repair", f"分镜 {shot_index} 修复完成", 88)
    return True


def _reconcat_repaired_video(
    summary: dict[str, Any],
    output_path: Path,
    shots: list[dict[str, Any]],
    flow_print: Callable[[str], None],
) -> None:
    clip_files: list[Path] = []
    ordered_shot_indices: list[int] = []
    for shot in sorted(shots, key=lambda item: int(item.get("shot_index", 0) or 0)):
        shot_index = int(shot.get("shot_index", 0) or 0)
        formal_clip = output_path / f"seedance_shot_{shot_index:02d}.mp4"
        local_scene_clip = output_path / f"seedance_shot_{shot_index:02d}_local_scene.mp4"
        selected_clip = formal_clip if formal_clip.exists() else local_scene_clip
        if selected_clip.exists():
            clip_files.append(selected_clip)
            ordered_shot_indices.append(shot_index)
    if not clip_files:
        return

    raw_path = output_path / "seedance_raw.mp4"
    final_path = output_path / "seedance_final.mp4"
    from agent.seedance_video_renderer import (
        _concat_videos,
        _overlay_storyboard_subtitles,
        _transition_types_for_clip_order,
    )

    shots_by_index = {int(shot.get("shot_index", 0)): shot for shot in shots}
    concat_result = _concat_videos(
        clip_files,
        raw_path,
        transition_types=_transition_types_for_clip_order(shots_by_index, ordered_shot_indices),
    )
    if not concat_result.get("success"):
        flow_print(
            "[video_generation_workflow] 修复后视频重新拼接失败："
            f"{concat_result.get('error', '未知错误')}"
        )
        summary["reconcat_error"] = concat_result.get("error", "未知错误")
        return

    try:
        sub_result = _overlay_storyboard_subtitles(
            source_video_path=raw_path,
            final_video_path=final_path,
            shots=shots,
        )
        if not sub_result.get("success"):
            shutil.copyfile(raw_path, final_path)
    except Exception:
        shutil.copyfile(raw_path, final_path)
    summary["reconcat_success"] = True
    flow_print(f"[video_generation_workflow] 修复后视频已重新合成：{final_path}")


def _record_skipped(summary: dict[str, Any], shot_index: int, repair_strategy: str, error: str) -> None:
    summary["skipped_count"] += 1
    summary["records"].append(
        {
            "shot_index": shot_index,
            "status": "skipped",
            "repair_strategy": repair_strategy,
            "error": error,
        }
    )


def _record_failed(summary: dict[str, Any], shot_index: int, repair_strategy: str, error: str) -> None:
    summary["failed_count"] += 1
    summary["records"].append(
        {
            "shot_index": shot_index,
            "status": "failed",
            "repair_strategy": repair_strategy,
            "error": error,
        }
    )
