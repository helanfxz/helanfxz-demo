"""
任务创建演示应用。

这个文件提供一个最小可运行的 FastAPI 页面，用来验证：

1. 前端表单输入是否符合预期
2. 文件上传是否成功进入后端
3. 任务创建模块能否正常返回结果

当前应用故意保持单文件，目标是先把链路跑通并便于审查。
"""

from __future__ import annotations

import builtins
from datetime import datetime, timezone
from html import escape
import logging
import os
from pathlib import Path
import traceback
import re
import signal
import subprocess
import threading
import time
from typing import List
from urllib.parse import unquote, urlparse
from uuid import uuid4
import json
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from commerce_style_templates import STYLE_TEMPLATES, style_template_by_id
from agent.asset_preprocessor import preprocess_all_assets
from agent.requirement_structurer import check_input_sufficiency, generate_chat_question
from agent.video_generation_workflow import continue_video_generation_workflow, run_video_generation_workflow
from video_task_module import (
    CreateVideoTaskCommand,
    InMemoryTaskRepository,
    TaskValidationError,
    UploadedAsset,
    confirm_task_primary_product_selections,
    create_video_task,
    fail_task_workflow,
    finish_task_workflow,
    approve_task_script_review,
    request_task_script_regeneration,
    start_task_workflow,
    update_task_primary_product_preflight,
    update_task_workflow_progress,
    update_task_workflow_partial,
    update_task_assets,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("task_creation_demo_app")
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
VERBOSE_LOG = os.getenv("AIGC_VERBOSE_LOG") == "1"


def print(*args, **kwargs):  # type: ignore[override]
    """默认隐藏细节调试输出，避免控制台被内部步骤刷屏。"""

    if VERBOSE_LOG:
        builtins.print(*args, **kwargs)


def _flow_print(message: str) -> None:
    """输出用户需要看到的关键流程日志。"""

    builtins.print(message, flush=True)

app = FastAPI(title="任务创建演示")
repository = InMemoryTaskRepository()
UPLOAD_ROOT = Path(".uploads")
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_ROOT)), name="uploads")
SERVER_PID = os.getpid()
SERVER_STARTED_AT = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
RUN_INSTANCE_ID = uuid4().hex[:8]
APP_FILE = os.path.abspath(__file__)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """返回任务创建演示页面。"""

    print(
        "[task_creation_demo_app] 访问首页，开始展示任务创建表单。"
        f" pid={SERVER_PID}, run_id={RUN_INSTANCE_ID}",
        flush=True,
    )
    logger.info("访问首页，开始展示任务创建表单。")
    return _html_response(_render_page())


@app.get("/api/tasks/{task_id}")
def task_detail_json(task_id: str):
    """返回任务状态的 JSON，供前端轮询。"""

    try:
        task = repository.get(task_id)
    except KeyError:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    task_dict = task.to_dict()
    return {
        "task_id": task_dict.get("task_id"),
        "status": task_dict.get("status"),
        "workflow_stage": task_dict.get("workflow_stage"),
        "workflow_message": task_dict.get("workflow_message"),
        "workflow_progress": int(task_dict.get("workflow_progress", 0) or 0),
        "workflow_events": task_dict.get("workflow_events", []),
        "workflow_result": task_dict.get("workflow_result") or {},
    }


@app.get("/api/health")
def api_health():
    """只读健康检查；仅暴露配置状态，不返回密钥或 endpoint 明文。"""

    return {
        "status": "ok",
        "server_pid": SERVER_PID,
        "run_instance_id": RUN_INSTANCE_ID,
        "started_at": SERVER_STARTED_AT,
        "port": int(os.getenv("PORT", "8010") or "8010"),
        "upload_root": str(UPLOAD_ROOT.resolve()),
        "disable_llm": os.getenv("AIGC_DISABLE_LLM") == "1",
        "disable_video_model": os.getenv("AIGC_DISABLE_VIDEO_MODEL") == "1",
        "ark_text_configured": bool(os.getenv("ARK_API_KEY") and os.getenv("ARK_TEXT_ENDPOINT_ID")),
        "ark_video_configured": bool(os.getenv("ARK_API_KEY") and os.getenv("ARK_VIDEO_ENDPOINT_ID")),
        "task_count": _repository_task_count(),
    }


@app.get("/tasks/{task_id}/report.json")
def task_report_json(task_id: str):
    """导出当前内存任务的复核报告，不触发任何生成流程。"""

    try:
        task = repository.get(task_id)
    except KeyError:
        return JSONResponse({"error": "任务不存在"}, status_code=404)

    task_dict = task.to_dict()
    workflow_result = task_dict.get("workflow_result") or {}
    task_summary = dict(task_dict)
    task_summary.pop("workflow_result", None)
    report = {
        "generated_at": _utc_iso_now(),
        "task": task_summary,
        "workflow_result": workflow_result,
        "artifact_dir": _artifact_dir_for_report(task_id, workflow_result),
        "video_urls": _video_urls_for_report(workflow_result),
    }
    return _json_compatible(report)


@app.get("/tasks/{task_id}", response_class=HTMLResponse)
def task_detail(task_id: str) -> str:
    """展示任务详情页。"""

    print(
        f"[task_creation_demo_app] 访问任务详情页：task_id={task_id}, "
        f"pid={SERVER_PID}, run_id={RUN_INSTANCE_ID}",
        flush=True,
    )
    logger.info("访问任务详情页：task_id=%s", task_id)

    try:
        task = repository.get(task_id)
    except KeyError as exc:
        print(f"[task_creation_demo_app] 任务详情页加载失败：{exc}", flush=True)
        logger.warning("任务详情页加载失败：%s", exc)
        return _html_response(_render_page(error_message=str(exc)))

    return _html_response(_render_page(success_task=task.to_dict(), page_mode="detail"))


@app.post("/tasks", response_class=HTMLResponse, response_model=None)
async def create_task_page(
    title: str = Form(...),
    selling_points: str = Form(...),
    target_platform: str = Form(...),
    duration_seconds: int = Form(...),
    style: str = Form(...),
    style_template_id: str = Form(""),
    custom_style_prompt: str = Form(""),
    product_type: str = Form(""),
    target_audience: str = Form(""),
    usage_scene: str = Form(""),
    creative_direction: str = Form(""),
    forbidden_changes: str = Form(""),
    chat_history: str = Form(""),
    video_urls: str = Form(""),
    files: List[UploadFile] = File(default_factory=list),
):
    """
    接收表单并创建任务。

    这里把卖点文本按行拆开，文件则只提取最小元数据。
    当前不保存文件内容，只验证上传动作是否和创建任务动作连通。
    """

    _flow_print(
        "[task_creation_demo_app] 收到任务创建请求："
        f"title={title}, target_platform={target_platform}, "
        f"duration_seconds={duration_seconds}, style={style}, "
        f"file_count={len(files)}"
    )
    logger.info(
        "收到任务创建请求：title=%s, target_platform=%s, duration_seconds=%s, style=%s, custom_style_prompt=%s, file_count=%s",
        title,
        target_platform,
        duration_seconds,
        style,
        custom_style_prompt,
        len(files),
    )

    uploaded_assets = [
        UploadedAsset(
            filename=file.filename or "unnamed",
            content_type=file.content_type or "application/octet-stream",
            asset_type=_asset_type_from_upload(file.content_type or "", file.filename or ""),
        )
        for file in files
        if file.filename
    ]
    link_assets = _video_link_assets(video_urls)
    uploaded_assets.extend(link_assets)
    print(
        "[task_creation_demo_app] 文件元数据提取完成："
        f"{[asset.filename for asset in uploaded_assets]}",
        flush=True,
    )
    logger.info("文件元数据提取完成：%s", [asset.filename for asset in uploaded_assets])

    # 把页面表单转换成任务模块能理解的命令对象。
    # 这里还不创建任务，只是完成“外部输入 -> 内部输入结构”的转换。
    style = _effective_style_value(style_template_id, style)
    custom_style_prompt = _compose_template_style_prompt(style_template_id, custom_style_prompt)
    command = CreateVideoTaskCommand(
        title=title,
        selling_points=_split_selling_points(selling_points),
        target_platform=target_platform,
        duration_seconds=duration_seconds,
        style=style,
        custom_style_prompt=custom_style_prompt,
        product_type=product_type,
        target_audience=target_audience,
        usage_scene=usage_scene,
        creative_direction=creative_direction,
        forbidden_changes=_split_selling_points(forbidden_changes),
        chat_history=_split_selling_points(chat_history),
        uploaded_assets=uploaded_assets,
    )
    print("[task_creation_demo_app] 创建任务命令对象完成，准备调用任务模块。", flush=True)
    logger.info("创建任务命令对象完成，准备调用任务创建模块。")

    try:
        # 创建任务实体，并写入当前的内存仓储。
        # 返回值 task 是后续工作流的统一入口，里面已经有 task_id。
        task = create_video_task(command, repository)
    except TaskValidationError as exc:
        _flow_print(f"[task_creation_demo_app] 任务创建失败：{exc}")
        logger.warning("任务创建失败：%s", exc)
        return _html_response(_render_page(error_message=str(exc), form_values={
            "title": title,
            "selling_points": selling_points,
            "target_platform": target_platform,
            "duration_seconds": str(duration_seconds),
            "style": style,
            "style_template_id": style_template_id,
            "custom_style_prompt": custom_style_prompt,
            "product_type": product_type,
            "target_audience": target_audience,
            "usage_scene": usage_scene,
            "creative_direction": creative_direction,
            "forbidden_changes": forbidden_changes,
            "chat_history": chat_history,
            "video_urls": video_urls,
        }))

    # 任务创建后才有 task_id，因此文件保存放在这里。
    # 保存后的 file_path 会继续进入多模态素材理解和本地视频预览。
    uploaded_assets = await _save_uploaded_files(task.task_id, files)
    uploaded_assets.extend(link_assets)
    task = update_task_assets(task.task_id, repository, uploaded_assets)

    # 素材预检可能包含图像分割/识别，不能阻塞表单提交请求。
    # 先跳转到详情页，再由后台线程完成预检；需要确认 SKU 时详情页会显示确认框。
    _start_background_preflight(task.task_id)
    return RedirectResponse(url=f"/tasks/{task.task_id}", status_code=303)


@app.post("/tasks/{task_id}/primary-product-confirmation", response_class=HTMLResponse, response_model=None)
async def confirm_primary_product_page(
    task_id: str,
    selections: List[str] = Form(default_factory=list),
):
    """保存用户选择的主商品候选，然后启动完整工作流。"""

    try:
        confirm_task_primary_product_selections(
            task_id,
            repository,
            _parse_primary_product_selections(selections),
        )
    except (KeyError, TaskValidationError) as exc:
        task = repository.get(task_id)
        return _html_response(_render_page(error_message=str(exc), success_task=task.to_dict(), page_mode="detail"))

    _flow_print(f"[task_creation_demo_app] 主商品确认完成：task_id={task_id}")
    _launch_task_workflow(task_id)
    return RedirectResponse(url=f"/tasks/{task_id}", status_code=303)


@app.post("/tasks/{task_id}/script-review/approve", response_class=HTMLResponse, response_model=None)
async def approve_script_review_page(task_id: str, request: Request):
    """保存用户编辑后的剧本分镜，然后继续后续视频生成。"""

    try:
        task = repository.get(task_id)
        form = await request.form()
        script_plan, storyboard, script_review_variants = _parse_script_review_submission(form, task.workflow_result)
        approve_task_script_review(
            task_id,
            repository,
            script_plan=script_plan,
            storyboard=storyboard,
            script_review_variants=script_review_variants,
            reviewer_note=str(form.get("reviewer_note", "")),
        )
        _start_background_approved_workflow(task_id)
    except (KeyError, TaskValidationError, ValueError) as exc:
        task = repository.get(task_id)
        return _html_response(_render_page(error_message=str(exc), success_task=task.to_dict(), page_mode="detail"))

    return RedirectResponse(url=f"/tasks/{task_id}", status_code=303)


@app.post("/tasks/{task_id}/script-review/regenerate", response_class=HTMLResponse, response_model=None)
async def regenerate_script_review_page(
    task_id: str,
    request: Request,
):
    """把用户对剧本的不通过意见追加给 LLM，重新生成剧本/分镜草稿。"""

    try:
        task = repository.get(task_id)
        form = await request.form()
        script_plan, storyboard, _script_review_variants = _parse_script_review_submission(form, task.workflow_result)
        feedback = _compose_regeneration_feedback(str(form.get("feedback", "")), script_plan, storyboard)
        request_task_script_regeneration(task_id, repository, feedback)
        _start_background_workflow(task_id)
    except (KeyError, TaskValidationError) as exc:
        task = repository.get(task_id)
        return _html_response(_render_page(error_message=str(exc), success_task=task.to_dict(), page_mode="detail"))

    return RedirectResponse(url=f"/tasks/{task_id}", status_code=303)


@app.post("/api/check_input")
async def check_input_api(
    title: str = Form(""),
    selling_points: str = Form(""),
    product_type: str = Form(""),
    target_audience: str = Form(""),
    usage_scene: str = Form(""),
    creative_direction: str = Form(""),
    chat_history: str = Form(""),
    file_count: int = Form(0),
):
    task_data = {
        "title": title,
        "selling_points": _split_selling_points(selling_points),
        "product_type": product_type,
        "target_audience": target_audience,
        "usage_scene": usage_scene,
        "creative_direction": creative_direction,
        "chat_history": _split_selling_points(chat_history),
        "uploaded_assets": [{"filename": "x"} for _ in range(file_count)] if file_count > 0 else [],
    }
    result = check_input_sufficiency(task_data)
    return result


@app.post("/api/chat_question")
async def chat_question_api(
    title: str = Form(""),
    selling_points: str = Form(""),
    product_type: str = Form(""),
    target_audience: str = Form(""),
    usage_scene: str = Form(""),
    creative_direction: str = Form(""),
    chat_history: str = Form(""),
    question_index: int = Form(0),
):
    task_data = {
        "title": title,
        "selling_points": _split_selling_points(selling_points),
        "product_type": product_type,
        "target_audience": target_audience,
        "usage_scene": usage_scene,
        "creative_direction": creative_direction,
    }
    result = generate_chat_question(task_data, _split_selling_points(chat_history), question_index)
    return result


def _split_selling_points(raw_text: str) -> List[str]:
    """把表单里的多值文本拆成列表，兼容旧版逗号拼接和新版逐行提交。"""

    return [
        item.strip()
        for item in re.split(r"[\n,，；;|]+", str(raw_text or ""))
        if item.strip()
    ]


def _compose_template_style_prompt(style_template_id: str, custom_style_prompt: str) -> str:
    """把用户选择的风格模板合并到现有补充说明里。"""

    template = style_template_by_id(style_template_id)
    template_prompt = str((template or {}).get("planning_prompt") or "").strip()
    user_prompt = str(custom_style_prompt or "").strip()
    if template_prompt and user_prompt:
        return f"{template_prompt}\n\n用户补充：{user_prompt}"
    return template_prompt or user_prompt


def _effective_style_value(style_template_id: str, style: str) -> str:
    """选中风格模板时，以模板 style_value 为唯一基础风格，避免模板和手选风格冲突。"""

    template = style_template_by_id(style_template_id)
    template_style = str((template or {}).get("style_value") or "").strip()
    return template_style or str(style or "").strip()


def _repository_task_count() -> int:
    """返回内存仓储任务数；只读诊断用，不改变仓储接口。"""

    tasks = getattr(repository, "_tasks", None)
    return len(tasks) if isinstance(tasks, dict) else 0


def _utc_iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artifact_dir_for_report(task_id: str, workflow_result: dict) -> str:
    artifact_dir = str((workflow_result or {}).get("artifacts_dir") or "").strip()
    if artifact_dir:
        return artifact_dir
    default_dir = UPLOAD_ROOT / task_id / "artifacts"
    return str(default_dir) if default_dir.exists() else ""


def _video_urls_for_report(workflow_result: dict) -> list[dict]:
    """提取 A/B 视频摘要，方便评审快速定位成片。"""

    if not isinstance(workflow_result, dict):
        return []
    items: list[dict] = []

    def add(label: str, result: dict) -> None:
        if not isinstance(result, dict):
            return
        video_url = _video_url_from_result(result)
        video_path = str(result.get("video_path") or "")
        if not video_url and not video_path:
            return
        items.append({
            "label": label,
            "success": bool(result.get("success")),
            "video_url": video_url,
            "video_path": video_path,
            "render_mode": result.get("render_mode", ""),
            "elapsed_seconds": result.get("elapsed_seconds", ""),
        })

    add("A_default", workflow_result.get("render_result") or {})
    ab_variants = workflow_result.get("ab_variants") or {}
    if isinstance(ab_variants, dict):
        for variant_id, variant in ab_variants.items():
            if not isinstance(variant, dict):
                continue
            variant_result = variant.get("render_result") or {}
            if not variant_result and variant.get("video_path"):
                variant_result = {"success": variant.get("success"), "video_path": variant.get("video_path")}
            add(str(variant_id), variant_result)
    return items


def _json_compatible(value):
    """把 Path、datetime 等对象转换成 JSONResponse 可安全编码的普通结构。"""

    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_compatible(item) for item in value]
    if isinstance(value, tuple):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _asset_type_from_upload(content_type: str, filename: str = "") -> str:
    """Return the coarse material type used by the workflow."""

    normalized = str(content_type or "").lower()
    if normalized.startswith("image/"):
        return "image"
    if normalized.startswith("video/"):
        return "video"
    suffix = Path(filename or "").suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
        return "image"
    if suffix in {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}:
        return "video"
    return "unknown"


def _video_link_assets(raw_text: str) -> List[UploadedAsset]:
    """Parse user-provided video/reference links into external video assets."""

    assets: List[UploadedAsset] = []
    for index, token in enumerate(re.split(r"[\s,，；;|]+", str(raw_text or "")), start=1):
        url = token.strip()
        if not url:
            continue
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        name = Path(unquote(parsed.path)).name or f"external_video_{index}"
        assets.append(
            UploadedAsset(
                filename=name,
                content_type="video/external",
                public_url=url,
                asset_type="video",
                source_url=url,
            )
        )
    return assets


async def _save_uploaded_files(task_id: str, files: List[UploadFile]) -> List[UploadedAsset]:
    """把上传文件保存到当前任务目录。"""

    task_upload_dir = UPLOAD_ROOT / task_id
    task_upload_dir.mkdir(parents=True, exist_ok=True)
    saved_assets: List[UploadedAsset] = []

    for index, upload_file in enumerate(files, start=1):
        if not upload_file.filename:
            continue

        safe_name = _safe_filename(upload_file.filename)
        saved_name = f"{index:02d}_{safe_name}"
        file_path = task_upload_dir / saved_name
        content = await upload_file.read()

        # 当前直接保存到本地目录；后续换对象存储时，只需要替换这里。
        file_path.write_bytes(content)

        asset = UploadedAsset(
            filename=upload_file.filename,
            content_type=upload_file.content_type or "application/octet-stream",
            file_path=str(file_path),
            public_url=f"/uploads/{task_id}/{saved_name}",
            file_size=len(content),
            asset_type=_asset_type_from_upload(upload_file.content_type or "", upload_file.filename),
        )
        saved_assets.append(asset)
        print(
            "[task_creation_demo_app] 上传文件已保存："
            f"filename={asset.filename}, file_path={asset.file_path}, size={asset.file_size}",
            flush=True,
        )

    return saved_assets


def _prepare_primary_product_preflight(task_id: str):
    """在启动工作流前预检图片，低置信度时暂停等待用户确认。"""

    task = repository.get(task_id)
    output_dir = UPLOAD_ROOT / task_id / "preflight"
    assets = [
        {
            "file_path": asset.file_path,
            "asset_type": asset.asset_type or _asset_type_from_upload(asset.content_type, asset.filename),
        }
        for asset in task.uploaded_assets
        if asset.file_path
    ]
    preprocess_results = preprocess_all_assets(assets, str(output_dir))
    profiles_by_path = {
        str(result.get("original_path", "")): dict(result.get("primary_product", {}))
        for result in preprocess_results
        if result.get("primary_product")
    }
    task = update_task_primary_product_preflight(task_id, repository, profiles_by_path)
    pending_count = sum(
        1 for asset in task.uploaded_assets if asset.primary_product.get("requires_user_confirmation")
    )
    _flow_print(
        "[task_creation_demo_app] 主商品预检完成："
        f"task_id={task_id}, confirmation_required={pending_count}"
    )
    return task


def _start_background_preflight(task_id: str) -> None:
    """后台执行素材预检，保证创建任务接口可以立即跳转详情页。"""

    thread = threading.Thread(
        target=_run_preflight_background,
        args=(task_id,),
        daemon=True,
        name=f"aigc-preflight-{task_id[-8:]}",
    )
    thread.start()


def _run_preflight_background(task_id: str) -> None:
    """预检完成后自动进入工作流，或停在主商品确认阶段。"""

    try:
        update_task_workflow_progress(task_id, repository, "preflight", "正在预检上传素材，确认主商品锚点。", 2)
        task = _prepare_primary_product_preflight(task_id)
        if _task_requires_primary_product_confirmation(task.to_dict()):
            return
        _launch_task_workflow(task_id)
    except Exception as exc:  # noqa: BLE001 - 后台线程必须保存错误，避免详情页无限等待。
        fail_task_workflow(task_id, repository, f"素材预检失败：{exc}")
        _flow_print(f"[task_creation_demo_app] 素材预检失败：task_id={task_id}, error={exc}")
        traceback.print_exc()


def _task_requires_primary_product_confirmation(task: dict) -> bool:
    """判断任务是否仍有需要用户确认的素材。"""

    return any(
        asset.get("primary_product", {}).get("requires_user_confirmation")
        for asset in task.get("uploaded_assets", [])
    )


def _parse_primary_product_selections(raw_selections: List[str]) -> dict[int, int]:
    """把前端提交的 `素材索引:候选索引` 转成任务模块需要的映射。"""

    selections: dict[int, int] = {}
    for raw_selection in raw_selections:
        try:
            asset_index, candidate_index = str(raw_selection).split(":", 1)
            selections[int(asset_index)] = int(candidate_index)
        except (TypeError, ValueError) as exc:
            raise TaskValidationError("主商品选择格式无效，请刷新页面后重新选择。") from exc
    return selections


def _public_script_variant_meta(variant_id: str) -> tuple[str, str]:
    """把内部方案 ID 转成用户能理解的名称和取舍说明。"""

    if variant_id == "B_ideal_commerce_scene":
        return (
            "方案 B：场景带货版",
            "更强调人物、环境和使用结果，剧情更完整；需要重点检查商品外观、logo 和结构是否稳定。",
        )
    return (
        "方案 A：稳妥保真版",
        "更重视上传商品的外观一致性，动作会更克制；适合先确认商品不跑偏。",
    )


def _safe_script_variant_key(variant_id: str) -> str:
    """生成可放进表单字段名的方案 key。"""

    safe_key = re.sub(r"[^A-Za-z0-9_]+", "_", str(variant_id or "")).strip("_")
    return safe_key or "A_conservative_fidelity"


def _script_review_variant_map(workflow_result: dict) -> dict[str, dict]:
    """统一整理剧本确认页可展示的方案，兼容旧任务的 A-only 结构。"""

    workflow_result = workflow_result or {}
    raw_variants = workflow_result.get("script_review_variants") or {}
    variants: dict[str, dict] = {}
    if isinstance(raw_variants, dict):
        for variant_id, variant in raw_variants.items():
            if isinstance(variant, dict):
                variants[str(variant_id)] = dict(variant)
    if "A_conservative_fidelity" not in variants:
        variants["A_conservative_fidelity"] = {
            "script_plan": workflow_result.get("script_plan", {}) or {},
            "storyboard": workflow_result.get("storyboard", []) or [],
            "readable_script": workflow_result.get("readable_script", {}) or {},
        }
    ordered: dict[str, dict] = {}
    for variant_id in ("A_conservative_fidelity", "B_ideal_commerce_scene"):
        variant = variants.get(variant_id)
        if not variant:
            continue
        label, description = _public_script_variant_meta(variant_id)
        variant.setdefault("label", label)
        variant.setdefault("description", description)
        variant.setdefault("script_plan", {})
        variant.setdefault("storyboard", [])
        variant.setdefault("readable_script", {})
        ordered[variant_id] = variant
    for variant_id, variant in variants.items():
        if variant_id in ordered:
            continue
        label, description = _public_script_variant_meta(variant_id)
        variant.setdefault("label", label)
        variant.setdefault("description", description)
        ordered[variant_id] = variant
    return ordered


def _clean_public_script_text(value: object, *, max_chars: int = 240) -> str:
    """清理内部字段里的技术措辞，避免确认页出现难以理解的系统文本。"""

    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return ""
    replacements = {
        "image_to_video": "图生视频",
        "text_to_video": "文生视频",
        "hard_cut": "直接切换",
        "continue_from_previous": "承接上一镜",
        "planner_source": "",
        "material_strategy": "",
        "selected_prompt_skill": "",
        "product_identity_card": "",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = re.sub(r"\b[A-Za-z_]{3,}:[A-Za-z0-9_:-]+\b", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ；;,，")
    if len(text) > max_chars:
        text = text[: max_chars - 1].rstrip() + "…"
    return text


def _shot_public_goal(shot: dict) -> str:
    return _clean_public_script_text(
        shot.get("scene_goal") or shot.get("purpose") or shot.get("narrative_role") or "",
        max_chars=180,
    )


def _shot_public_action(shot: dict) -> str:
    """优先使用更像自然语言画面描述的字段，避免只显示保守修正后的短 action。"""

    candidates = [
        shot.get("acting_direction"),
        shot.get("visual_description"),
        shot.get("video_prompt"),
        shot.get("action"),
    ]
    for candidate in candidates:
        text = _clean_public_script_text(candidate, max_chars=260)
        if text and not _looks_like_internal_prompt(text):
            return text
    return _clean_public_script_text(shot.get("action", ""), max_chars=220)


def _looks_like_internal_prompt(text: str) -> bool:
    internal_markers = ("{", "}", "required_for_variant", "render_strategy", "forbidden_variation")
    return any(marker in text for marker in internal_markers)


def _script_variant_body_text(script_plan: dict) -> str:
    body = script_plan.get("body", [])
    if isinstance(body, str):
        return body.strip()
    if isinstance(body, (list, tuple)):
        return "\n".join(str(item) for item in body if str(item).strip())
    return ""


def _style_template_preview_url(template: dict) -> str:
    filename = str(template.get("preview_filename") or "").strip()
    if not filename:
        return ""
    return f"/uploads/style_templates/{filename}"


def _render_style_template_cards(selected_template_id: str = "") -> str:
    cards: list[str] = []
    for template in STYLE_TEMPLATES:
        template_id = str(template.get("id", ""))
        preview_url = _style_template_preview_url(template)
        selected = template_id == str(selected_template_id or "")
        beats = "".join(
            f"<li>{escape(str(beat))}</li>"
            for beat in template.get("beats", [])
        )
        preview_html = (
            (
                f'<video src="{escape(preview_url)}" controls autoplay muted loop playsinline '
                f'preload="metadata" onclick="event.stopPropagation()" '
                f'aria-label="{escape(str(template.get("title", "风格模板")))}样例视频"></video>'
                '<span class="template-video-badge">样例视频</span>'
            )
            if preview_url
            else '<div class="template-preview-placeholder">样片待补充</div>'
        )
        cards.append(
            f"""
            <article
              class="trend-template-card{' selected' if selected else ''}"
              data-template-id="{escape(template_id)}"
              data-template-title="{escape(str(template.get('title', '')))}"
              data-style="{escape(str(template.get('style_value', 'product_showcase')))}"
              onclick="selectTrendTemplate(this)"
            >
              <div class="template-preview">{preview_html}</div>
              <div class="template-body">
                <div class="template-title-row">
                  <strong>{escape(str(template.get('title', '风格模板')))}</strong>
                  <span>选择</span>
                </div>
                <p>{escape(str(template.get('tagline', '')))}</p>
                <em>适合：{escape(str(template.get('best_for', '多类商品')))}</em>
                <ul>{beats}</ul>
              </div>
            </article>
            """
        )
    return f"""
      <section class="trend-template-section">
        <div class="template-section-head">
          <div>
            <h3>选择带货风格模板</h3>
            <p>模板会影响剧本结构和表达方式；系统仍会根据你上传的素材重新适配，不会生硬套用样片商品。</p>
          </div>
          <span>可选</span>
        </div>
        <input type="hidden" name="style_template_id" id="f-style-template" value="{escape(str(selected_template_id or ''))}">
        <input type="hidden" id="f-style-template-label" value="">
        <div class="trend-template-grid">{"".join(cards)}</div>
      </section>
    """


def _render_script_variant_timeline(storyboard: list[dict]) -> str:
    """渲染轻量分镜时间轴，让确认页更接近课题要求里的分镜可视化。"""

    if not storyboard:
        return ""
    total = sum(_safe_int_form_value(shot.get("duration_seconds"), 3) for shot in storyboard)
    cursor = 0
    items: list[str] = []
    for index, shot in enumerate(storyboard):
        duration = _safe_int_form_value(shot.get("duration_seconds"), 3)
        start = cursor
        cursor += duration
        width = max(12, round(duration / max(total, 1) * 100, 1))
        label = str(shot.get("subtitle") or shot.get("scene_goal") or shot.get("purpose") or f"分镜 {index + 1}")
        items.append(
            f"""
            <div class="script-timeline-item" style="flex-basis:{width}%">
              <strong>{escape(str(start))}-{escape(str(cursor))}s</strong>
              <span>{escape(_clean_public_script_text(label, max_chars=34))}</span>
            </div>
            """
        )
    return f"""
    <div class="script-timeline" aria-label="分镜时间轴">
      <div class="script-timeline-head">
        <strong>分镜时间轴</strong>
        <span>总时长约 {escape(str(total))} 秒</span>
      </div>
      <div class="script-timeline-track">{"".join(items)}</div>
    </div>
    """


def _parse_script_review_submission(form, workflow_result: dict) -> tuple[dict, list[dict], dict[str, dict]]:
    """把 A/B 剧本确认页的两套可编辑方案还原为工作流输入。"""

    edited_variants = _parse_script_review_variants_form(form, workflow_result)
    if edited_variants:
        primary_variant = edited_variants.get("A_conservative_fidelity") or next(iter(edited_variants.values()))
        return (
            dict(primary_variant.get("script_plan") or {}),
            list(primary_variant.get("storyboard") or []),
            edited_variants,
        )

    script_plan, storyboard = _parse_flat_script_review_form(form, workflow_result)
    return script_plan, storyboard, {}


def _parse_script_review_form(form, workflow_result: dict) -> tuple[dict, list[dict]]:
    """兼容测试和旧调用：返回确认后默认主视频使用的剧本与分镜。"""

    script_plan, storyboard, _edited_variants = _parse_script_review_submission(form, workflow_result)
    return script_plan, storyboard


def _parse_script_review_variants_form(form, workflow_result: dict) -> dict[str, dict]:
    """解析确认页上同时展示的 A/B 可编辑剧本，不再依赖单选选择。"""

    variant_map = _script_review_variant_map(workflow_result or {})
    form_keys = list(getattr(form, "keys", lambda: [])())
    edited_variants: dict[str, dict] = {}
    for variant_id, variant in variant_map.items():
        field_prefix = f"variant_{_safe_script_variant_key(variant_id)}__"
        if not any(str(key).startswith(field_prefix) for key in form_keys):
            continue
        script_plan, storyboard = _parse_single_script_review_variant(
            form,
            workflow_result or {},
            variant_id=variant_id,
            variant=variant,
            field_prefix=field_prefix,
        )
        label, description = _public_script_variant_meta(variant_id)
        edited_variants[variant_id] = {
            **dict(variant),
            "label": str(variant.get("label") or label),
            "description": str(variant.get("description") or description),
            "script_plan": script_plan,
            "storyboard": storyboard,
            "user_editable": True,
            "user_confirmed": True,
        }
    return edited_variants


def _parse_single_script_review_variant(
    form,
    workflow_result: dict,
    *,
    variant_id: str,
    variant: dict,
    field_prefix: str,
) -> tuple[dict, list[dict]]:
    """解析某一个 A/B 方案的字段。"""

    original_script = dict(variant.get("script_plan") or workflow_result.get("script_plan") or {})
    original_storyboard = list(variant.get("storyboard") or workflow_result.get("storyboard") or [])

    def field_value(name: str, fallback_name: str | None = None, default: str = "") -> str:
        return str(form.get(f"{field_prefix}{name}", form.get(fallback_name or name, default))).strip()

    script_plan = {
        **original_script,
        "rich_story_text": field_value("script_synopsis", "script_synopsis"),
        "hook": field_value("script_hook", "script_hook"),
        "body": _split_selling_points(field_value("script_body", "script_body")),
        "cta": field_value("script_cta", "script_cta"),
        "review_variant_id": variant_id,
    }
    script_plan.pop("selected_review_variant", None)
    script_plan.pop("selected_review_variant_label", None)
    if not script_plan["rich_story_text"] and not script_plan["hook"] and not script_plan["body"] and not script_plan["cta"]:
        raise TaskValidationError("剧本内容不能为空。")

    try:
        shot_count_raw = form.get(f"{field_prefix}shot_count", form.get("shot_count"))
        shot_count = int(str(shot_count_raw or "0"))
    except ValueError as exc:
        raise TaskValidationError("分镜数量无效，请刷新页面后重试。") from exc

    storyboard: list[dict] = []
    for index in range(shot_count):
        original = dict(original_storyboard[index]) if index < len(original_storyboard) else {}
        shot = {
            **original,
            "shot_index": int(original.get("shot_index", index + 1) or index + 1),
            "duration_seconds": _safe_int_form_value(
                form.get(f"{field_prefix}shot_{index}_duration", form.get(f"shot_{index}_duration")),
                original.get("duration_seconds", 3),
            ),
            "scene_goal": field_value(f"shot_{index}_scene_goal", f"shot_{index}_scene_goal"),
            "action": field_value(f"shot_{index}_action", f"shot_{index}_action"),
            "subtitle": field_value(f"shot_{index}_subtitle", f"shot_{index}_subtitle"),
            "review_variant_id": variant_id,
        }
        shot.pop("selected_review_variant", None)
        if not shot["scene_goal"] and not shot["action"] and not shot["subtitle"]:
            continue
        storyboard.append(shot)

    if not storyboard:
        raise TaskValidationError("至少保留一个分镜。")
    return script_plan, storyboard


def _parse_flat_script_review_form(form, workflow_result: dict) -> tuple[dict, list[dict]]:
    """兼容早期单剧本确认表单。"""

    variant_map = _script_review_variant_map(workflow_result or {})
    selected_variant = variant_map.get("A_conservative_fidelity") or {}
    return _parse_single_script_review_variant(
        form,
        workflow_result or {},
        variant_id=str(selected_variant.get("variant_id") or "A_conservative_fidelity"),
        variant=selected_variant,
        field_prefix="",
    )


def _compose_regeneration_feedback(feedback: str, script_plan: dict, storyboard: list[dict]) -> str:
    """把用户反馈和当前表单编辑内容合并，确保重新生成能看到刚改过的字段。"""

    parts = []
    cleaned_feedback = str(feedback or "").strip()
    if cleaned_feedback:
        parts.append(f"用户重新生成意见：{cleaned_feedback}")
    synopsis = str(script_plan.get("rich_story_text", "")).strip()
    if synopsis:
        parts.append(f"用户当前编辑的总剧本：{synopsis}")
    if script_plan.get("hook"):
        parts.append(f"用户当前编辑的开场：{script_plan.get('hook')}")
    if script_plan.get("body"):
        parts.append("用户当前编辑的卖点展开：" + "；".join(str(item) for item in script_plan.get("body", [])))
    if script_plan.get("cta"):
        parts.append(f"用户当前编辑的结尾：{script_plan.get('cta')}")
    for shot in storyboard:
        shot_index = shot.get("shot_index", "")
        goal = str(shot.get("scene_goal", "")).strip()
        action = str(shot.get("action", "")).strip()
        subtitle = str(shot.get("subtitle", "")).strip()
        parts.append(
            f"用户当前编辑的分镜{shot_index}：目标={goal or '未填'}；画面怎么拍={action or '未填'}；字幕={subtitle or '未填'}"
        )
    combined = "\n".join(part for part in parts if part.strip()).strip()
    if not combined:
        raise TaskValidationError("请填写希望修改的方向，或先编辑剧本/分镜内容。")
    return combined


def _safe_int_form_value(value, fallback: int) -> int:
    try:
        parsed = int(str(value or fallback))
    except (TypeError, ValueError):
        parsed = int(fallback or 3)
    return max(1, min(15, parsed))


def _launch_task_workflow(task_id: str) -> None:
    """推进任务状态并启动后台工作流。"""

    print(f"[task_creation_demo_app] 准备自动启动工作流：task_id={task_id}", flush=True)
    logger.info("准备自动启动工作流：task_id=%s", task_id)
    start_task_workflow(task_id, repository)
    _start_background_workflow(task_id)
    _flow_print(f"[task_creation_demo_app] 任务已提交后台工作流：task_id={task_id}")
    logger.info("任务已提交后台工作流：task_id=%s", task_id)


def _start_background_workflow(task_id: str) -> None:
    """启动后台工作流线程，避免用户在创建任务接口里等待视频生成。"""

    thread = threading.Thread(
        target=_run_workflow_background,
        args=(task_id,),
        daemon=True,
        name=f"aigc-workflow-{task_id[-8:]}",
    )
    thread.start()


def _start_background_approved_workflow(task_id: str) -> None:
    """启动用户确认剧本后的渲染线程。"""

    thread = threading.Thread(
        target=_run_approved_workflow_background,
        args=(task_id,),
        daemon=True,
        name=f"aigc-approved-{task_id[-8:]}",
    )
    thread.start()


def _run_workflow_background(task_id: str) -> None:
    """后台执行规划阶段，并在剧本/分镜可审阅后暂停。"""

    try:
        task = repository.get(task_id)

        def progress_callback(stage: str, message: str, progress: int, partial: dict | None = None) -> None:
            update_task_workflow_progress(task_id, repository, stage, message, progress)
            if partial:
                update_task_workflow_partial(task_id, repository, partial)

        workflow_result = run_video_generation_workflow(
            task.to_dict(),
            progress_callback=progress_callback,
            stop_after_plan_review=True,
        )
        finish_task_workflow(task_id, repository, workflow_result)
        _flow_print(f"[task_creation_demo_app] 后台工作流执行完成：task_id={task_id}")
    except Exception as exc:  # noqa: BLE001 - 后台线程需要兜底保存错误，否则前端只能一直显示 processing。
        fail_task_workflow(task_id, repository, f"后台工作流执行失败：{exc}")
        _flow_print(f"[task_creation_demo_app] 后台工作流执行失败：task_id={task_id}, error={exc}")
        traceback.print_exc()


def _run_approved_workflow_background(task_id: str) -> None:
    """用户确认剧本后继续执行渲染、A/B 候选和内容审核。"""

    try:
        task = repository.get(task_id)

        def progress_callback(stage: str, message: str, progress: int, partial: dict | None = None) -> None:
            update_task_workflow_progress(task_id, repository, stage, message, progress)
            if partial:
                update_task_workflow_partial(task_id, repository, partial)

        workflow_result = continue_video_generation_workflow(
            task.to_dict(),
            progress_callback=progress_callback,
        )
        finish_task_workflow(task_id, repository, workflow_result)
        _flow_print(f"[task_creation_demo_app] 用户确认后工作流执行完成：task_id={task_id}")
    except Exception as exc:  # noqa: BLE001
        fail_task_workflow(task_id, repository, f"确认后工作流执行失败：{exc}")
        _flow_print(f"[task_creation_demo_app] 确认后工作流执行失败：task_id={task_id}, error={exc}")
        traceback.print_exc()


def _safe_filename(filename: str) -> str:
    """生成适合本地保存的文件名。"""

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename.strip())
    return safe_name or "upload.bin"


def _render_page(
    error_message: str | None = None,
    success_task: dict | None = None,
    form_values: dict | None = None,
    page_mode: str = "create",
) -> str:
    """渲染页面。"""

    values = form_values or {}
    title = escape(values.get("title", ""))
    selling_points = escape(values.get("selling_points", ""))
    target_platform = escape(values.get("target_platform", "tiktok"))
    duration_seconds = escape(values.get("duration_seconds", "15"))
    style = escape(values.get("style", "product_showcase"))
    style_template_id = escape(values.get("style_template_id", ""))
    custom_style_prompt = escape(values.get("custom_style_prompt", ""))
    product_type = escape(values.get("product_type", ""))
    target_audience = escape(values.get("target_audience", ""))
    usage_scene = escape(values.get("usage_scene", ""))
    creative_direction = escape(values.get("creative_direction", ""))
    forbidden_changes = escape(values.get("forbidden_changes", ""))
    chat_history = escape(values.get("chat_history", ""))
    video_urls = escape(values.get("video_urls", ""))
    # --- new render below ---

    error_banner = ""
    if error_message:
        error_banner = f'<div class="toast toast-error"><span>⚠</span>{escape(error_message)}</div>'

    task_id_js = "null"
    is_detail_processing = False
    page_body = ""

    if success_task:
        task_id_raw = success_task.get("task_id", "")
        task_id_js = json.dumps(task_id_raw, ensure_ascii=False)
        is_detail_processing = success_task.get("status") in {"queued", "processing"}
        is_waiting_primary_product = success_task.get("workflow_stage") == "primary_product_confirmation"
        workflow_progress = int(success_task.get("workflow_progress", 0) or 0)
        workflow_result = success_task.get("workflow_result", {})
        uploaded_assets = success_task.get("uploaded_assets", [])

        status_map = {
            "completed": ("done", "生成完成"),
            "needs_review": ("warn", "待确认"),
            "failed": ("error", "生成失败"),
            "processing": ("active", "生成中"),
            "queued": ("active", "队列中"),
        }
        cur_status = success_task.get("status", "processing")
        dot_cls, status_label = status_map.get(cur_status, ("active", cur_status))
        if is_waiting_primary_product:
            dot_cls, status_label = "warn", "待确认主商品"
        status_message = (
            str(success_task.get("workflow_message", "") or "请确认主商品后继续生成。")
            if is_waiting_primary_product
            else str(success_task.get("workflow_message", "") or "正在初始化…")
        )
        waiting_note = (
            '<p class="hint">确认主商品后，系统才会启动工作流。</p>'
            if is_waiting_primary_product
            else ""
        )

        selling_pt_tags = "".join(
            f'<span class="sp-tag">{escape(p)}</span>'
            for p in success_task.get("selling_points", [])
        ) or '<span class="sp-tag muted">未填写</span>'

        asset_thumbs = ""
        for a in uploaded_assets:
            fpath = a.get("file_path", "")
            pub_url = a.get("public_url") or f'/uploads/{task_id_raw}/{escape(a["filename"])}'
            asset_type = str(a.get("asset_type") or "").lower()
            if fpath and any(fpath.lower().endswith(x) for x in (".jpg", ".jpeg", ".png", ".webp")):
                asset_thumbs += f'<img src="{escape(pub_url)}" alt="{escape(a.get('filename',''))}" class="thumb">'
            elif asset_type == "video" and str(pub_url).startswith("/uploads/"):
                asset_thumbs += f'<video src="{escape(pub_url)}" class="thumb" muted preload="metadata"></video>'
            elif asset_type == "video":
                asset_thumbs += '<div class="thumb-placeholder">视频链接</div>'
            else:
                asset_thumbs += f'<div class="thumb-placeholder">{escape(a.get("filename","?")[:6])}</div>'

        primary_confirmation_html = _render_primary_product_confirmation(success_task)
        script_review_html = _render_script_review_panel(success_task)
        workflow_detail = _render_workflow_result(workflow_result)
        stage_panel_html = _render_stage_panel(success_task, success_task.get("workflow_events", []))
        report_link = (
            f'<a href="/tasks/{escape(task_id_raw)}/report.json" target="_blank" rel="noopener">导出任务报告</a>'
            if task_id_raw
            else ""
        )
        current_stage = str(success_task.get("workflow_stage", ""))
        if current_stage == "primary_product_confirmation":
            main_content = f"""
    {error_banner}
    {primary_confirmation_html}
    <div id="status-box" class="status-banner status-{dot_cls}">
      <span class="status-pulse {dot_cls}" id="status-dot"></span>
      <span id="status-text" class="status-text">{escape(status_message)}</span>
    </div>
    {waiting_note}
            """
        elif current_stage == "script_review":
            main_content = f"""
    {error_banner}
    <div id="status-box" class="status-banner status-{dot_cls}">
      <span class="status-pulse {dot_cls}" id="status-dot"></span>
      <span id="status-text" class="status-text">{escape(status_message)}</span>
    </div>
    {script_review_html}
            """
        elif success_task.get("status") in {"queued", "processing"}:
            main_content = f"""
    {error_banner}
    <div id="status-box" class="status-banner status-{dot_cls}">
      <span class="status-pulse {dot_cls}" id="status-dot"></span>
      <span id="status-text" class="status-text">{escape(status_message)}</span>
    </div>
    {_render_generation_progress_card(success_task)}
            """
        else:
            main_content = f"""
    {error_banner}
    <div id="status-box" class="status-banner status-{dot_cls}">
      <span class="status-pulse {dot_cls}" id="status-dot"></span>
      <span id="status-text" class="status-text">{escape(status_message)}</span>
    </div>
    <div class="workflow-output-grid" id="workflow-output">{workflow_detail}</div>
            """

        page_body = f"""
<div class="result-layout">
  <!-- Left: progress rail -->
  <aside class="progress-rail">
    <div class="rail-header">
      <div class="rail-dot {dot_cls}"></div>
      <span class="rail-label" id="rail-label">{status_label}</span>
    </div>
    <div class="rail-arc" id="rail-arc">
      <svg viewBox="0 0 44 44" class="arc-svg">
        <circle cx="22" cy="22" r="18" class="arc-bg"/>
        <circle cx="22" cy="22" r="18" class="arc-fill" id="arc-fill"
          stroke-dasharray="{round(113.1 * workflow_progress / 100, 1)} 113.1"/>
      </svg>
      <span class="arc-pct" id="arc-pct">{workflow_progress}%</span>
    </div>
    <div class="rail-actions">
      <button type="button" onclick="copyCurrentTaskLink(this)">复制任务链接</button>
      {report_link}
      <small>可关闭页面后继续查看</small>
    </div>
    <div class="stage-rail" id="stage-rail">{stage_panel_html}</div>
    <div class="asset-preview">
      <p class="rail-section-label">已上传素材</p>
      <div class="thumb-row">{asset_thumbs if asset_thumbs else '<span class="muted-sm">无素材</span>'}</div>
    </div>
    <div class="sp-preview">
      <p class="rail-section-label">卖点</p>
      <div class="sp-tags">{selling_pt_tags}</div>
    </div>
    <a href="/" class="new-task-btn">＋ 新建任务</a>
  </aside>

  <!-- Center: main content -->
  <main class="result-main">
    {main_content}
  </main>
</div>
"""
    else:
        # ── Create form ──────────────────────────────────────────────
        page_body = f"""
{error_banner}
<div class="create-layout">
  <form id="main-form" action="/tasks" method="post" enctype="multipart/form-data" novalidate onsubmit="return checkSufficiencyAndSubmit(event)">

    <!-- Step 1: Upload -->
    <div class="step-card active" id="step-1">
      <div class="step-header">
        <span class="step-num">1</span>
        <div>
          <h2>上传商品图片 / 视频素材</h2>
          <p class="step-desc">支持商品图片、商品视频片段，也可以填写公开视频或素材链接作为参考。</p>
        </div>
      </div>
      <div class="upload-zone" id="upload-zone" onclick="document.getElementById('file-input').click()">
        <div class="upload-icon">↑</div>
        <p class="upload-hint">点击或拖拽图片 / 视频到这里</p>
        <p class="upload-sub">图片用于商品外观锚定，视频用于动作、场景和使用方式参考</p>
        <input id="file-input" name="files" type="file" multiple accept="image/*,video/*" style="display:none" onchange="handleFiles(this)">
      </div>
      <div class="preview-grid" id="preview-grid"></div>
      <label class="field-label" style="margin-top:18px">视频链接
        <p class="field-hint">每行一个链接。当前会作为外部视频参考素材记录；直链视频和公开视频页面链接都可以先填入。</p>
        <textarea name="video_urls" rows="3" placeholder="https://example.com/product-demo.mp4">{video_urls}</textarea>
      </label>
      <div class="step-footer">
        <button type="button" class="btn-next" onclick="goStep(2)">下一步 →</button>
      </div>
    </div>

    <!-- Step 2: Product info -->
    <div class="step-card" id="step-2">
      <div class="step-header">
        <span class="step-num">2</span>
        <div>
          <h2>商品信息</h2>
          <p class="step-desc">帮助 AI 准确理解你的商品，越详细效果越好。</p>
        </div>
      </div>

      <label class="field-label">商品名称 <span class="req">*</span>
        <input name="title" id="f-title" value="{title}" placeholder="例如：Redmi Book Pro 15 轻薄笔记本" required>
      </label>

      <div class="field-label">核心卖点 <span class="req">*</span>
        <p class="field-hint">选择或自己填写，最多 5 条，每条建议 8-20 字</p>
        <div class="preset-tags" id="preset-tags">
          <button type="button" class="preset-tag" onclick="addPreset(this)">超薄轻巧，随时携带</button>
          <button type="button" class="preset-tag" onclick="addPreset(this)">长续航，全天不焦虑</button>
          <button type="button" class="preset-tag" onclick="addPreset(this)">高清大屏，视觉震撼</button>
          <button type="button" class="preset-tag" onclick="addPreset(this)">高性能处理器，流畅不卡顿</button>
          <button type="button" class="preset-tag" onclick="addPreset(this)">精致做工，品质感拉满</button>
          <button type="button" class="preset-tag" onclick="addPreset(this)">学生党性价比首选</button>
          <button type="button" class="preset-tag" onclick="addPreset(this)">办公神器，效率翻倍</button>
          <button type="button" class="preset-tag" onclick="addPreset(this)">快速充电，30分钟满电</button>
          <button type="button" class="preset-tag" onclick="addPreset(this)">颜值在线，多色可选</button>
          <button type="button" class="preset-tag" onclick="addPreset(this)">大容量，存储不焦虑</button>
        </div>
        <div class="sp-input-list" id="sp-list"></div>
        <button type="button" class="add-sp-btn" onclick="addSPInput()">＋ 自定义卖点</button>
        <textarea name="selling_points" id="sp-hidden" style="display:none">{selling_points}</textarea>
      </div>

      <label class="field-label">商品类型
        <div class="chip-group" id="product-type-chips">
          <input type="hidden" name="product_type" id="f-product-type" value="{product_type}">
          <button type="button" class="chip" data-val="笔记本" onclick="selectChip(this,'f-product-type')">💻 笔记本</button>
          <button type="button" class="chip" data-val="手机" onclick="selectChip(this,'f-product-type')">📱 手机</button>
          <button type="button" class="chip" data-val="耳机" onclick="selectChip(this,'f-product-type')">🎧 耳机</button>
          <button type="button" class="chip" data-val="水杯" onclick="selectChip(this,'f-product-type')">☕ 水杯</button>
          <button type="button" class="chip" data-val="服饰" onclick="selectChip(this,'f-product-type')">👕 服饰</button>
          <button type="button" class="chip" data-val="美妆" onclick="selectChip(this,'f-product-type')">💄 美妆</button>
          <button type="button" class="chip" data-val="家居" onclick="selectChip(this,'f-product-type')">🏠 家居</button>
          <button type="button" class="chip" data-val="其他" onclick="selectChip(this,'f-product-type')">📦 其他</button>
        </div>
        <input id="product-type-custom" class="custom-requirement-input" data-custom-target="f-product-type" data-custom-mode="single" value="" placeholder="自定义商品类型，例如：可折叠磁吸露营灯" oninput="syncCustomRequirement(this)">
      </label>

      <div class="step-footer two">
        <button type="button" class="btn-back" onclick="goStep(1)">← 上一步</button>
        <button type="button" class="btn-next" onclick="goStep(3)">下一步 →</button>
      </div>
    </div>

    <!-- Step 3: Style & audience -->
    <div class="step-card" id="step-3">
      <div class="step-header">
        <span class="step-num">3</span>
        <div>
          <h2>风格与受众</h2>
          <p class="step-desc">告诉 AI 你想要什么感觉的视频，以及卖给谁。</p>
        </div>
      </div>

      <input type="hidden" name="target_platform" value="tiktok">
      <input type="hidden" name="duration_seconds" value="15">

      {_render_style_template_cards(style_template_id)}

      <label class="field-label">视频风格
        <input type="hidden" name="style" id="f-style" value="{style if style else 'product_showcase'}">
        <div class="style-cards" id="style-cards">
          <div class="style-card" data-val="product_showcase" onclick="selectStyle(this)">
            <div class="style-icon">🎬</div>
            <strong>商品展示</strong>
            <p>专注展示外观与细节，简洁有力</p>
          </div>
          <div class="style-card" data-val="lifestyle" onclick="selectStyle(this)">
            <div class="style-icon">☀️</div>
            <strong>生活场景</strong>
            <p>融入真实生活，建立使用代入感</p>
          </div>
          <div class="style-card" data-val="premium" onclick="selectStyle(this)">
            <div class="style-icon">✨</div>
            <strong>高质感</strong>
            <p>暗调光影，强调品质与高端感</p>
          </div>
          <div class="style-card" data-val="energetic" onclick="selectStyle(this)">
            <div class="style-icon">⚡</div>
            <strong>活力节奏</strong>
            <p>快节奏剪辑，适合年轻用户群</p>
          </div>
        </div>
      </label>

      <label class="field-label">使用场景（可选）
        <input type="hidden" name="usage_scene" id="f-scene" value="{usage_scene}">
        <div class="chip-group">
          <button type="button" class="chip" data-val="办公桌" onclick="toggleChip(this,'f-scene')">🖥 办公桌</button>
          <button type="button" class="chip" data-val="通勤" onclick="toggleChip(this,'f-scene')">🚇 通勤</button>
          <button type="button" class="chip" data-val="宿舍" onclick="toggleChip(this,'f-scene')">🛏 宿舍</button>
          <button type="button" class="chip" data-val="咖啡店" onclick="toggleChip(this,'f-scene')">☕ 咖啡店</button>
          <button type="button" class="chip" data-val="居家" onclick="toggleChip(this,'f-scene')">🏠 居家</button>
          <button type="button" class="chip" data-val="户外" onclick="toggleChip(this,'f-scene')">🌿 户外</button>
        </div>
        <input id="scene-custom" class="custom-requirement-input" data-custom-target="f-scene" data-custom-mode="multi" value="" placeholder="自定义使用场景，例如：夜间帐篷内照明、车尾收纳" oninput="syncCustomRequirement(this)">
      </label>

      <label class="field-label">目标人群（可选）
        <input type="hidden" name="target_audience" id="f-audience" value="{target_audience}">
        <div class="chip-group">
          <button type="button" class="chip" data-val="学生" onclick="toggleChip(this,'f-audience')">🎓 学生</button>
          <button type="button" class="chip" data-val="职场人" onclick="toggleChip(this,'f-audience')">💼 职场人</button>
          <button type="button" class="chip" data-val="游戏玩家" onclick="toggleChip(this,'f-audience')">🎮 游戏玩家</button>
          <button type="button" class="chip" data-val="设计师" onclick="toggleChip(this,'f-audience')">🎨 设计师</button>
          <button type="button" class="chip" data-val="礼物购买者" onclick="toggleChip(this,'f-audience')">🎁 礼物</button>
          <button type="button" class="chip" data-val="泛人群" onclick="toggleChip(this,'f-audience')">👥 泛人群</button>
        </div>
        <input id="audience-custom" class="custom-requirement-input" data-custom-target="f-audience" data-custom-mode="multi" value="" placeholder="自定义目标人群，例如：周末露营新手、车主" oninput="syncCustomRequirement(this)">
      </label>

      <label class="field-label">创作方向（可选）
        <input type="hidden" name="creative_direction" id="f-direction" value="{creative_direction}">
        <div class="chip-group">
          <button type="button" class="chip" data-val="保守展示" onclick="selectChip(this,'f-direction')">📐 稳健展示</button>
          <button type="button" class="chip" data-val="轻剧情种草" onclick="selectChip(this,'f-direction')">🌱 轻剧情种草</button>
          <button type="button" class="chip" data-val="功能演示" onclick="selectChip(this,'f-direction')">⚙️ 功能演示</button>
          <button type="button" class="chip" data-val="促销转化" onclick="selectChip(this,'f-direction')">🛒 促销转化</button>
        </div>
      </label>

      <label class="field-label">补充说明（可选）
        <textarea name="custom_style_prompt" id="custom-style-prompt" placeholder="例如：画面要有夜晚氛围，突出屏幕发光效果，节奏稍慢一些。" rows="3">{custom_style_prompt}</textarea>
      </label>

      <!-- hidden fields -->
      <input type="hidden" name="forbidden_changes" value="{forbidden_changes}">
      <input type="hidden" name="chat_history" id="chat-history" value="">

      <div class="step-footer two">
        <button type="button" class="btn-back" onclick="goStep(2)">← 上一步</button>
        <button type="button" class="btn-next" onclick="goStep(4)">下一步 →</button>
      </div>
    </div>

    <!-- Step 4: Review & submit -->
    <div class="step-card" id="step-4">
      <div class="step-header">
        <span class="step-num">4</span>
        <div>
          <h2>确认并生成剧本</h2>
          <p class="step-desc">先生成可审阅剧本和分镜，通过后再进入视频生成。</p>
        </div>
      </div>

      <div class="review-grid" id="review-grid">
        <!-- filled by JS -->
      </div>

      <!-- AI 需求对话 -->
      <details class="chat-details" open>
        <summary>💬 和 AI 聊聊更多想法（可选）</summary>
        <div class="chat-body" id="chat-body">
          <div class="chat-bubble bot">我已了解商品基础信息。有什么特别想法？比如想突出什么氛围、避开什么效果，都可以告诉我。</div>
        </div>
        <div class="chat-input-row">
          <input type="text" id="chat-input" placeholder="随便说说…" onkeydown="if(event.key==='Enter')sendChat()">
          <button type="button" onclick="sendChat()">发送</button>
        </div>
      </details>

      <div class="submit-area">
        <button type="submit" class="btn-submit" id="btn-submit">
          <span class="submit-icon">▶</span> 生成剧本
        </button>
        <p class="hint">生成过程中可关闭页面，稍后通过链接查看结果</p>
      </div>

      <div class="step-footer two">
        <button type="button" class="btn-back" onclick="goStep(3)">← 上一步</button>
        <div></div>
      </div>
    </div>

  </form>

  <!-- Right: feature panel -->
  <aside class="feature-aside">
    <div class="feature-hero">
      <div class="feature-logo">AI</div>
      <h3>先审剧本<br>再生成视频</h3>
      <p>上传商品图，填写卖点，先产出可编辑剧本和分镜</p>
    </div>
    <div class="feature-list">
      <div class="feature-row">
        <span class="feat-icon">🔍</span>
        <div>
          <strong>商品身份卡</strong>
          <p>多模态分析固化品牌标识、外观特征，全流程复用，不依赖重复描述</p>
        </div>
      </div>
      <div class="feature-row">
        <span class="feat-icon">🎯</span>
        <div>
          <strong>模板直传架构</strong>
          <p>品牌信息零损耗注入视频提示词，消除多跳 LLM 信息递减问题</p>
        </div>
      </div>
      <div class="feature-row">
        <span class="feat-icon">🛡</span>
        <div>
          <strong>三层确定性拦截</strong>
          <p>渲染策略、Logo保真、可拍性审核全部由规则决定，不依赖模型随机</p>
        </div>
      </div>
      <div class="feature-row">
        <span class="feat-icon">🔄</span>
        <div>
          <strong>修复后验证闭环</strong>
          <p>AI 审视关键帧 → 规则驱动修复 → 二次抽帧验证，形成真正闭环</p>
        </div>
      </div>
    </div>
    <div class="step-indicator" id="step-indicator">
      <div class="si-dot active" data-step="1">1</div>
      <div class="si-line"></div>
      <div class="si-dot" data-step="2">2</div>
      <div class="si-line"></div>
      <div class="si-dot" data-step="3">3</div>
      <div class="si-line"></div>
      <div class="si-dot" data-step="4">4</div>
    </div>
  </aside>
</div>
"""

    return _build_full_page(page_body, task_id_js, is_detail_processing)





_CUSTOM_REQUIREMENT_FIELD_CSS = """
.custom-requirement-input{margin-top:8px}
.script-review-panel{background:#fff;border:1px solid #e8eaf0;border-radius:12px;padding:22px;box-shadow:0 12px 30px rgba(15,23,42,.06)}
.focused-script-page{max-width:1180px;margin:0 auto}
.script-review-head{display:flex;justify-content:space-between;gap:16px;align-items:flex-start;margin-bottom:18px}
.script-review-head h3{font-size:22px;margin-bottom:6px;color:#111827}
.script-review-head p{font-size:14px;color:#6b7280;line-height:1.6}
.script-review-head span{background:#fef3c7;color:#92400e;border-radius:999px;padding:4px 10px;font-size:12px;font-weight:700;white-space:nowrap}
.script-review-action-bar{position:sticky;top:10px;z-index:5;display:flex;align-items:center;justify-content:space-between;gap:14px;background:#111827;color:#fff;border-radius:12px;padding:14px 16px;margin-bottom:18px;box-shadow:0 16px 30px rgba(15,23,42,.18)}
.script-review-action-bar strong{display:block;font-size:14px;margin-bottom:3px}
.script-review-action-bar span{display:block;font-size:12px;color:#cbd5e1;line-height:1.4}
.script-review-action-buttons{display:flex;gap:10px;flex-shrink:0}
.script-review-action-buttons button{white-space:nowrap}
.script-review-action-buttons .btn-submit{max-width:none;width:auto;padding:11px 18px;font-size:14px}
.script-review-form label,.script-regenerate-form label{display:block;font-size:13px;font-weight:600;color:#374151;margin-bottom:12px}
.script-review-form textarea,.script-regenerate-form textarea{margin-top:6px}
.script-variant-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-bottom:18px}
.script-variant-card{border:1px solid #e5e7eb;border-radius:14px;background:#fbfcff;padding:14px}
.script-variant-card:has(input[type=radio]:checked){border-color:#6c63ff;box-shadow:0 0 0 3px rgba(108,99,255,.12);background:#fff}
.script-variant-choice{display:flex!important;gap:10px;align-items:flex-start;background:#fff;border:1px solid #e8eaf0;border-radius:12px;padding:12px;margin-bottom:14px!important;cursor:pointer}
.script-variant-choice input{width:18px;height:18px;margin-top:2px;accent-color:#6c63ff}
.script-variant-choice strong{display:block;font-size:16px;color:#111827;margin-bottom:4px}
.script-variant-choice em{display:block;font-style:normal;font-size:12px;font-weight:500;color:#6b7280;line-height:1.55}
.script-main-card{background:#f8fafc;border:1px solid #e5e7eb;border-radius:12px;padding:16px;margin-bottom:18px}
.script-main-card textarea{background:#fff;line-height:1.65}
.script-timeline{background:#fff;border:1px solid #e8eaf0;border-radius:10px;padding:12px;margin:4px 0 14px}
.script-timeline-head{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:10px}
.script-timeline-head strong{font-size:13px;color:#111827}
.script-timeline-head span{font-size:12px;color:#6b7280}
.script-timeline-track{display:flex;gap:6px;align-items:stretch}
.script-timeline-item{min-width:90px;background:#eef2ff;border:1px solid #dfe4ff;border-radius:8px;padding:8px;overflow:hidden}
.script-timeline-item strong{display:block;font-size:11px;color:#4f46e5;margin-bottom:4px}
.script-timeline-item span{display:block;font-size:12px;color:#374151;line-height:1.35;word-break:break-word}
.script-line-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.script-section-title{font-size:15px;color:#111827;margin:6px 0 12px}
.script-shot-grid{display:grid;grid-template-columns:1fr;gap:10px;margin:14px 0}
.script-shot-grid.compact{gap:12px}
.script-shot-card{border:1px solid #e8eaf0;border-radius:10px;background:#fff;padding:14px}
.script-shot-head{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:10px}
.script-shot-head label{max-width:120px;margin:0}
.script-review-actions{display:flex;justify-content:flex-end;gap:10px;margin-top:14px}
.script-regenerate-form{border-top:1px solid #e8eaf0;margin-top:18px;padding-top:16px}
.trend-template-section{border:1px solid #e8eaf0;background:#f8fafc;border-radius:14px;padding:16px;margin-bottom:18px}
.template-section-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;margin-bottom:12px}
.template-section-head h3{font-size:16px;color:#111827;margin-bottom:4px}
.template-section-head p{font-size:12px;color:#6b7280;line-height:1.55}
.template-section-head span{background:#eef2ff;color:#4f46e5;border-radius:999px;padding:4px 9px;font-size:11px;font-weight:700;white-space:nowrap}
.trend-template-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}
.trend-template-card{border:2px solid #e5e7eb;background:#fff;border-radius:12px;overflow:hidden;cursor:pointer;transition:all .15s}
.trend-template-card:hover{border-color:#a5b4fc;box-shadow:0 10px 24px rgba(15,23,42,.06)}
.trend-template-card.selected{border-color:#6c63ff;box-shadow:0 0 0 3px rgba(108,99,255,.12)}
.trend-template-card.template-overridden{opacity:.62}
.trend-template-card.template-overridden .template-title-row span{background:#f3f4f6;color:#6b7280}
.template-preview{position:relative;aspect-ratio:9/16;background:#111827;overflow:hidden}
.template-preview video{width:100%;height:100%;object-fit:cover;display:block}
.template-video-badge{position:absolute;left:10px;top:10px;background:rgba(17,24,39,.82);color:#fff;border-radius:999px;padding:4px 8px;font-size:11px;font-weight:700;pointer-events:none}
.template-preview-placeholder{height:100%;display:flex;align-items:center;justify-content:center;color:#9ca3af;font-size:12px;background:linear-gradient(135deg,#111827,#374151)}
.template-body{padding:12px}
.template-title-row{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:6px}
.template-title-row strong{font-size:14px;color:#111827}
.template-title-row span{background:#f3f4f6;color:#374151;border-radius:999px;padding:3px 8px;font-size:11px;font-weight:700}
.trend-template-card.selected .template-title-row span{background:#6c63ff;color:#fff}
.template-body p{font-size:12px;color:#374151;line-height:1.5;margin-bottom:6px}
.template-body em{display:block;font-style:normal;font-size:11px;color:#6b7280;line-height:1.45;margin-bottom:8px}
.template-body ul{padding-left:16px;font-size:11px;color:#4b5563;line-height:1.55}
.generation-progress-panel{background:#fff;border:1px solid #e8eaf0;border-radius:12px;padding:20px;max-width:760px;margin:0 auto}
.generation-progress-head{display:flex;justify-content:space-between;gap:18px;align-items:flex-start;margin-bottom:14px}
.generation-progress-head h3{font-size:20px;margin-bottom:6px;color:#111827}
.generation-progress-head p{font-size:14px;color:#6b7280;line-height:1.6}
.generation-progress-head strong{font-size:24px;color:#6c63ff}
.progress-note{margin-top:6px;color:#4b5563!important;font-size:13px!important}
.progress-bar{height:10px;border-radius:999px;background:#edf0f5;overflow:hidden;margin-bottom:18px}
.progress-bar span{display:block;height:100%;background:#6c63ff;border-radius:999px;transition:width .3s}
.stage-checklist{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-bottom:16px}
.stage-check-item{border:1px solid #e5e7eb;background:#f9fafb;border-radius:10px;padding:10px}
.stage-check-item strong{display:block;font-size:12px;color:#111827;margin-bottom:3px}
.stage-check-item span{font-size:11px;color:#6b7280}
.stage-check-item.done{border-color:#bbf7d0;background:#f0fdf4}
.stage-check-item.active{border-color:#c7d2fe;background:#eef2ff}
.result-section.compact{padding:14px;margin-bottom:0}
.event-list{list-style:none;padding-left:0}
.video-comparison-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:16px}
.video-variant-card{background:#fff;border:1px solid #e8eaf0;border-radius:12px;padding:14px}
.video-variant-card h3{font-size:14px;margin-bottom:8px}
.video-variant-card video{width:100%;border-radius:10px;background:#000;max-height:420px}
.video-export-actions{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:10px}
.video-export-actions a{background:#111827;color:#fff;border-radius:8px;padding:7px 10px;font-size:12px;font-weight:700}
.video-export-actions a:nth-child(2){background:#f3f4f6;color:#374151}
.video-export-actions span{font-size:12px;color:#6b7280}
.video-variant-card ul{padding-left:18px;margin-top:8px;font-size:12px;color:#6b7280;line-height:1.6}
.progress-rail{background:#1a1a2e;color:#fff;border-radius:16px;padding:24px;position:sticky;top:20px;align-self:start}
.rail-actions{background:#23233d;border:1px solid #333454;border-radius:10px;padding:10px;margin:12px 0}
.rail-actions button,.rail-actions a{display:block;width:100%;background:#fff;color:#1a1a2e;border-radius:8px;padding:8px 10px;font-size:12px;font-weight:700;text-align:center;margin-bottom:8px}
.rail-actions a{background:#6c63ff;color:#fff}
.rail-actions small{display:block;color:#9ca3af;font-size:11px;margin-top:6px;text-align:center}
.stage-item{display:flex;gap:8px;align-items:flex-start;color:#9ca3af;font-size:12px;padding:4px 0}
.stage-dot{width:8px;height:8px;border-radius:50%;background:#4a4a6a;margin-top:5px;flex-shrink:0}
.stage-item.done .stage-dot{background:#22c55e}
.stage-item.active .stage-dot{background:#6c63ff;box-shadow:0 0 0 3px rgba(108,99,255,.18)}
.stage-item.failed .stage-dot{background:#ef4444}
.stage-info{display:flex;flex-direction:column;gap:2px;min-width:0}
.stage-label{color:#d1d5db}
.stage-label em{font-style:normal;color:#6b7280;margin-left:4px}
.stage-sub{font-size:11px;color:#9ca3af;line-height:1.35}
.stage-sub.retrying{color:#fbbf24}
.stage-sub.error{color:#fca5a5}
@media(max-width:980px){.script-variant-grid,.trend-template-grid{grid-template-columns:1fr}}
@media(max-width:800px){.video-comparison-grid,.script-line-grid,.stage-checklist{grid-template-columns:1fr}.script-review-head,.generation-progress-head{flex-direction:column}.script-timeline-track{overflow-x:auto}.progress-rail{display:none}}
"""


_CUSTOM_REQUIREMENT_FIELD_SCRIPT = r"""
function splitRequirementText(text){
  return String(text||'').split(/[，,；;|\n]+/).map(v=>v.trim()).filter(Boolean);
}
function uniqueVals(vals){
  const out=[];
  vals.forEach(v=>{ if(v&&!out.includes(v)) out.push(v); });
  return out;
}
function selectedChipValues(hiddenId){
  const h=document.getElementById(hiddenId);
  if(!h) return [];
  const grp=h.closest('.chip-group');
  if(!grp) return [];
  return Array.from(grp.querySelectorAll('.chip.selected')).map(c=>c.dataset.val).filter(Boolean);
}
function syncRequirementField(hiddenId){
  const h=document.getElementById(hiddenId);
  if(!h) return;
  const custom=document.querySelector('.custom-requirement-input[data-custom-target="'+hiddenId+'"]');
  const mode=custom?custom.dataset.customMode:'single';
  const customVals=custom?splitRequirementText(custom.value):[];
  if(mode==='multi'){
    h.value=uniqueVals(selectedChipValues(hiddenId).concat(customVals)).join(',');
  }else{
    h.value=customVals.length?customVals.join('、'):(selectedChipValues(hiddenId)[0]||'');
  }
}
function syncAllRequirementFields(){
  ['f-product-type','f-scene','f-audience','f-direction'].forEach(syncRequirementField);
}
function selectChip(btn,hiddenId){
  const grp=btn.closest('.chip-group');
  grp.querySelectorAll('.chip').forEach(c=>c.classList.remove('selected'));
  btn.classList.add('selected');
  const custom=document.querySelector('.custom-requirement-input[data-custom-target="'+hiddenId+'"]');
  if(custom) custom.value='';
  syncRequirementField(hiddenId);
}
function toggleChip(btn,hiddenId){
  btn.classList.toggle('selected');
  syncRequirementField(hiddenId);
}
function syncCustomRequirement(input){
  const target=input.dataset.customTarget;
  if(!target) return;
  if(input.dataset.customMode==='single'&&input.value.trim()){
    const h=document.getElementById(target);
    const grp=h?h.closest('.chip-group'):null;
    if(grp) grp.querySelectorAll('.chip').forEach(c=>c.classList.remove('selected'));
  }
  syncRequirementField(target);
}
function selectTrendTemplate(card){
  document.querySelectorAll('.trend-template-card').forEach(c=>c.classList.remove('selected','template-overridden'));
  card.classList.add('selected');
  document.querySelectorAll('.trend-template-card .template-title-row span').forEach(s=>s.textContent='选择');
  const templateId=card.dataset.templateId||'';
  const templateTitle=card.dataset.templateTitle||templateId;
  const style=card.dataset.style||'product_showcase';
  const hidden=document.getElementById('f-style-template');
  if(hidden) hidden.value=templateId;
  const labelHidden=document.getElementById('f-style-template-label');
  if(labelHidden) labelHidden.value=templateTitle;
  const styleHidden=document.getElementById('f-style');
  if(styleHidden) styleHidden.value=style;
  document.querySelectorAll('.style-card').forEach(c=>c.classList.toggle('selected',c.dataset.val===style));
}
function selectStyle(card){
  document.querySelectorAll('.style-card').forEach(c=>c.classList.remove('selected'));
  card.classList.add('selected');
  const h=document.getElementById('f-style');
  if(h) h.value=card.dataset.val;
  const templateHidden=document.getElementById('f-style-template');
  if(templateHidden&&templateHidden.value){
    templateHidden.value='';
    const labelHidden=document.getElementById('f-style-template-label');
    if(labelHidden) labelHidden.value='';
    document.querySelectorAll('.trend-template-card').forEach(c=>{
      c.classList.remove('selected');
      c.classList.add('template-overridden');
      const badge=c.querySelector('.template-title-row span');
      if(badge) badge.textContent='未选择';
    });
  }
}
function restoreChips(){
  document.querySelectorAll('.chip-group').forEach(grp=>{
    const hidden=grp.querySelector('input[type="hidden"]');
    if(!hidden||!hidden.value) return;
    const custom=document.querySelector('.custom-requirement-input[data-custom-target="'+hidden.id+'"]');
    const isMulti=custom&&custom.dataset.customMode==='multi';
    const vals=splitRequirementText(hidden.value);
    const matched=[];
    grp.querySelectorAll('.chip').forEach(c=>{
      if(vals.includes(c.dataset.val)){ c.classList.add('selected'); matched.push(c.dataset.val); }
    });
    const unmatched=vals.filter(v=>!matched.includes(v)&&v!=='其他');
    if(custom&&unmatched.length) custom.value=isMulti?unmatched.join('，'):unmatched.join('、');
    syncRequirementField(hidden.id);
  });
}
function syncSP(){
  const vals=Array.from(document.querySelectorAll('#sp-list .sp-input-item input'))
    .map(i=>i.value.trim()).filter(Boolean);
  const h=document.getElementById('sp-hidden');
  if(h) h.value=vals.join('\n');
}
function buildReview(){
  syncSP();
  syncAllRequirementFields();
  const fields=[
    ['商品名称','f-title'],['商品类型','f-product-type'],['使用场景','f-scene'],
    ['目标人群','f-audience'],['创作方向','f-direction'],['带货模板','f-style-template-label'],
  ];
  const g=document.getElementById('review-grid');if(!g) return;
  g.innerHTML='';
  fields.forEach(([label,id])=>{
    const el=document.getElementById(id);const val=el?el.value:'';
    if(!val) return;
    const d=document.createElement('div');d.className='review-item';
    d.innerHTML='<span class="review-key">'+label+'</span><span class="review-val">'+val+'</span>';
    g.appendChild(d);
  });
  const sp=document.getElementById('sp-hidden');
  if(sp&&sp.value){
    const d=document.createElement('div');d.className='review-item';
    d.innerHTML='<span class="review-key">核心卖点</span><span class="review-val">'+sp.value.replace(/\n/g,'，')+'</span>';
    g.appendChild(d);
  }
}
function checkSufficiencyAndSubmit(e){
  syncSP();
  syncAllRequirementFields();
  const title=document.getElementById('f-title');
  if(!title||!title.value.trim()){e.preventDefault();alert('请填写商品名称');goStep(2);return false;}
  const sp=document.getElementById('sp-hidden');
  if(!sp||!sp.value.trim()){e.preventDefault();alert('请至少添加一个卖点');goStep(2);return false;}
  const btn=document.getElementById('btn-submit');
  if(btn){btn.disabled=true;btn.innerHTML='<span class="submit-icon">...</span> 正在提交，请稍候';}
  return true;
}
function copyCurrentTaskLink(btn){
  const text=window.location.href;
  const done=()=>{ if(btn){ const old=btn.textContent; btn.textContent='已复制'; setTimeout(()=>{btn.textContent=old||'复制任务链接';},1400); } };
  if(navigator.clipboard&&navigator.clipboard.writeText){
    navigator.clipboard.writeText(text).then(done).catch(()=>{ window.prompt('复制任务链接', text); });
  }else{
    window.prompt('复制任务链接', text);
  }
}
"""


_TASK_POLLING_SCRIPT = r"""
function startTaskPolling(){
  if(!TASK_ID || !IS_PROCESSING) return;
  let completedReloaded=false;
  async function pollTask(){
    try{
      const resp=await fetch('/api/tasks/'+encodeURIComponent(TASK_ID),{cache:'no-store'});
      if(!resp.ok) return;
      const data=await resp.json();
      const status=String(data.status||'');
      const stage=String(data.workflow_stage||'');
      if(status==='queued'||status==='processing'){
        window.setTimeout(()=>window.location.reload(),2500);
        return;
      }
      if(!completedReloaded&&(status==='completed'||status==='failed'||status==='needs_review'||stage==='script_review')){
        completedReloaded=true;
        window.location.reload();
      }
    }catch(err){
      window.setTimeout(()=>window.location.reload(),5000);
    }
  }
  window.setTimeout(pollTask,2500);
}
startTaskPolling();
"""


def _build_full_page(body: str, task_id_js: str, is_detail_processing: bool) -> str:
    """组装完整 HTML 页面：CSS + shell + body + JS。"""

    page = (
        _PAGE_TEMPLATE.replace('const TASK_ID = "%%TASK_ID%%";', "const TASK_ID = %%TASK_ID%%;")
        .replace("%%BODY%%", body)
        .replace("%%TASK_ID%%", task_id_js)
        .replace("%%IS_PROCESSING%%", "true" if is_detail_processing else "false")
    )
    page = page.replace("</style>", f"{_CUSTOM_REQUIREMENT_FIELD_CSS}\n</style>", 1)
    page = page.replace(
        "function handleFiles(input){\n"
        "  const grid=document.getElementById('preview-grid');\n"
        "  grid.innerHTML='';\n"
        "  Array.from(input.files).forEach(f=>{\n"
        "    const img=document.createElement('img');\n"
        "    img.className='preview-thumb';\n"
        "    img.src=URL.createObjectURL(f);\n"
        "    grid.appendChild(img);\n"
        "  });\n"
        "}",
        "function handleFiles(input){\n"
        "  const grid=document.getElementById('preview-grid');\n"
        "  grid.innerHTML='';\n"
        "  Array.from(input.files).forEach(f=>{\n"
        "    if(String(f.type||'').startsWith('video/')){\n"
        "      const v=document.createElement('video');\n"
        "      v.className='preview-thumb';v.controls=true;v.muted=true;v.src=URL.createObjectURL(f);\n"
        "      grid.appendChild(v);return;\n"
        "    }\n"
        "    const img=document.createElement('img');\n"
        "    img.className='preview-thumb';\n"
        "    img.src=URL.createObjectURL(f);\n"
        "    grid.appendChild(img);\n"
        "  });\n"
        "}",
        1,
    )
    page = page.replace(
        "\nrestoreChips();\nrestoreStyle();",
        f"\n{_TASK_POLLING_SCRIPT}\nrestoreChips();\nrestoreStyle();",
        1,
    )
    page = page.replace("\nrestoreChips();\nrestoreStyle();", f"\n{_CUSTOM_REQUIREMENT_FIELD_SCRIPT}\nrestoreChips();\nrestoreStyle();", 1)
    return page


_PAGE_TEMPLATE = '<!doctype html>\n<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width,initial-scale=1">\n<title>AIGC 带货视频生成</title>\n<style>\n*{box-sizing:border-box;margin:0;padding:0}\nbody{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f4f5f7;color:#1a1a2e;min-height:100vh}\na{color:inherit;text-decoration:none}\ninput,textarea,select{font-family:inherit;font-size:14px;border:1px solid #dde1e7;border-radius:8px;padding:10px 12px;width:100%;outline:none;background:#fff;transition:border .15s}\ninput:focus,textarea:focus{border-color:#6c63ff;box-shadow:0 0 0 3px rgba(108,99,255,.12)}\ntextarea{resize:vertical}\nbutton{font-family:inherit;cursor:pointer;border:none;border-radius:8px;padding:10px 20px;font-size:14px;transition:all .15s}\n\n/* ── Layout ── */\n.create-layout{display:grid;grid-template-columns:1fr 320px;gap:24px;max-width:1100px;margin:32px auto;padding:0 16px}\n.result-layout{display:grid;grid-template-columns:260px 1fr;gap:24px;max-width:1200px;margin:32px auto;padding:0 16px}\n@media(max-width:800px){.create-layout,.result-layout{grid-template-columns:1fr}.feature-aside,.result-aside{display:none}}\n\n/* ── Step cards ── */\n.step-card{background:#fff;border-radius:16px;padding:28px;border:1px solid #e8eaf0;display:none}\n.step-card.active{display:block}\n.step-header{display:flex;align-items:flex-start;gap:14px;margin-bottom:24px}\n.step-num{flex-shrink:0;width:36px;height:36px;border-radius:50%;background:#6c63ff;color:#fff;display:flex;align-items:center;justify-content:center;font-size:16px;font-weight:700}\n.step-header h2{font-size:18px;font-weight:700;margin-bottom:4px}\n.step-desc{font-size:13px;color:#6b7280}\n.step-footer{display:flex;justify-content:flex-end;margin-top:24px}\n.step-footer.two{justify-content:space-between}\n.btn-next{background:#6c63ff;color:#fff;padding:11px 28px;font-size:14px;font-weight:600;border-radius:10px}\n.btn-next:hover{background:#574fd6}\n.btn-back{background:#f3f4f6;color:#374151;padding:11px 20px;font-size:14px;border-radius:10px}\n.btn-back:hover{background:#e5e7eb}\n\n/* ── Upload zone ── */\n.upload-zone{border:2px dashed #c4c9d4;border-radius:12px;padding:40px 20px;text-align:center;cursor:pointer;transition:all .2s;background:#fafbfc}\n.upload-zone:hover,.upload-zone.drag{border-color:#6c63ff;background:#f0eeff}\n.upload-icon{font-size:32px;margin-bottom:10px;color:#6c63ff}\n.upload-hint{font-weight:600;color:#374151;margin-bottom:4px}\n.upload-sub{font-size:12px;color:#9ca3af}\n.preview-grid{display:flex;flex-wrap:wrap;gap:10px;margin-top:14px}\n.preview-thumb{width:80px;height:80px;object-fit:cover;border-radius:8px;border:2px solid #e5e7eb}\n\n/* ── Form fields ── */\n.field-label{display:block;font-size:13px;font-weight:600;color:#374151;margin-bottom:16px}\n.field-label input,.field-label textarea{margin-top:6px}\n.req{color:#ef4444}\n.chip-group{display:flex;flex-wrap:wrap;gap:8px;margin-top:8px}\n.chip{padding:6px 14px;border-radius:20px;background:#f3f4f6;color:#374151;font-size:13px;border:1.5px solid transparent;cursor:pointer}\n.chip.selected,.chip:hover{background:#ede9ff;border-color:#6c63ff;color:#6c63ff}\n.sp-input-row{display:flex;gap:8px;margin-top:8px}\n.sp-input-row input{flex:1}\n.sp-input-row button{flex-shrink:0;background:#6c63ff;color:#fff;border-radius:8px;padding:0 16px}\n.sp-tags-live{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}\n.sp-tag{background:#ede9ff;color:#6c63ff;border-radius:16px;padding:4px 12px;font-size:13px;display:flex;align-items:center;gap:6px}\n.sp-tag button{background:none;color:#6c63ff;font-size:14px;padding:0;line-height:1}\n.preset-tags{display:flex;flex-wrap:wrap;gap:7px;margin-top:8px}\n.preset-tag{padding:6px 13px;border-radius:18px;background:#f3f4f6;color:#374151;font-size:13px;border:1.5px solid transparent;cursor:pointer;transition:all .15s}\n.preset-tag:hover{background:#ede9ff;border-color:#6c63ff;color:#6c63ff}\n.preset-tag.used{background:#e5e7eb;color:#9ca3af;cursor:default;text-decoration:line-through}\n.sp-input-list{display:flex;flex-direction:column;gap:6px;margin-top:10px}\n.sp-input-item{display:flex;gap:6px;align-items:center}\n.sp-input-item input{flex:1}\n.sp-input-item button{flex-shrink:0;background:#fee2e2;color:#b91c1c;border-radius:6px;padding:6px 10px;font-size:13px}\n.add-sp-btn{margin-top:8px;background:#f3f4f6;color:#374151;border-radius:8px;padding:7px 14px;font-size:13px;border:1.5px dashed #d1d5db}\n.add-sp-btn:hover{background:#ede9ff;border-color:#6c63ff;color:#6c63ff}\n.field-hint{font-size:12px;color:#9ca3af;font-weight:400;margin-bottom:6px}\n.style-cards{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:8px}\n.style-card{border:2px solid #e8eaf0;border-radius:12px;padding:14px;cursor:pointer;transition:all .15s;background:#fff}\n.style-card:hover{border-color:#6c63ff;background:#faf9ff}\n.style-card.selected{border-color:#6c63ff;background:#ede9ff}\n.style-icon{font-size:22px;margin-bottom:6px}\n.style-card strong{display:block;font-size:14px;margin-bottom:3px;color:#1a1a2e}\n.style-card p{font-size:12px;color:#6b7280;line-height:1.4}\n\n/* ── Feature aside (create page) ── */\n.feature-aside{background:#1a1a2e;color:#fff;border-radius:16px;padding:28px;position:sticky;top:20px}\n.feature-logo{width:44px;height:44px;border-radius:12px;background:#6c63ff;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:18px;margin-bottom:14px}\n.feature-hero h3{font-size:20px;font-weight:700;line-height:1.3;margin-bottom:8px}\n.feature-hero p{font-size:13px;color:#9ca3af;margin-bottom:24px}\n.feature-row{display:flex;align-items:flex-start;gap:12px;margin-bottom:16px}\n.feat-icon{font-size:20px;flex-shrink:0;margin-top:2px}\n.feature-row strong{display:block;font-size:14px;margin-bottom:3px}\n.feature-row p{font-size:12px;color:#9ca3af;line-height:1.5}\n.step-indicator{display:flex;align-items:center;gap:0;margin-top:28px}\n.si-dot{width:28px;height:28px;border-radius:50%;background:#2d2d4a;border:2px solid #4a4a6a;color:#6b7280;font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center;transition:all .2s;flex-shrink:0}\n.si-dot.active{background:#6c63ff;border-color:#6c63ff;color:#fff}\n.si-dot.done{background:#22c55e;border-color:#22c55e;color:#fff}\n.si-line{flex:1;height:2px;background:#2d2d4a}\n\n/* ── Result aside ── */\n.result-aside{background:#1a1a2e;color:#fff;border-radius:16px;padding:24px;position:sticky;top:20px;align-self:start}\n.rail-header{display:flex;align-items:center;gap:10px;margin-bottom:16px}\n.rail-dot{width:10px;height:10px;border-radius:50%;background:#9ca3af}\n.rail-dot.running{background:#6c63ff;animation:pulse 1.4s infinite}\n.rail-dot.done{background:#22c55e}\n.rail-dot.error{background:#ef4444}\n@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}\n.rail-label{font-size:13px;font-weight:600}\n.rail-arc{display:flex;align-items:center;justify-content:center;margin:12px 0;position:relative}\n.arc-svg{width:80px;height:80px}\n.arc-bg{fill:none;stroke:#2d2d4a;stroke-width:4}\n.arc-fill{fill:none;stroke:#6c63ff;stroke-width:4;stroke-linecap:round;transform:rotate(-90deg);transform-origin:50% 50%;transition:stroke-dasharray .4s}\n.arc-pct{position:absolute;font-size:15px;font-weight:700}\n.stage-rail{display:flex;flex-direction:column;gap:6px;margin:12px 0}\n.rail-stage{font-size:12px;color:#9ca3af;padding:4px 0}\n.rail-stage.active{color:#6c63ff;font-weight:600}\n.rail-stage.done{color:#22c55e}\n.rail-section-label{font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px}\n.thumb-row{display:flex;flex-wrap:wrap;gap:6px}\n.thumb-row img{width:48px;height:48px;object-fit:cover;border-radius:6px;border:2px solid #2d2d4a}\n.muted-sm{font-size:12px;color:#6b7280}\n.sp-preview{margin-top:14px}\n.sp-tags{display:flex;flex-wrap:wrap;gap:5px}\n.sp-tag-sm{background:#2d2d4a;color:#9ca3af;border-radius:12px;padding:3px 10px;font-size:11px}\n.new-task-btn{display:block;text-align:center;background:#6c63ff;color:#fff;border-radius:10px;padding:10px;font-size:13px;font-weight:600;margin-top:20px}\n.new-task-btn:hover{background:#574fd6}\n\n/* ── Result main ── */\n.result-main{display:flex;flex-direction:column;gap:16px}\n.status-banner{display:flex;align-items:center;gap:10px;background:#fff;border-radius:12px;padding:14px 18px;border:1px solid #e8eaf0}\n.status-banner.status-running{border-left:3px solid #6c63ff}\n.status-banner.status-done{border-left:3px solid #22c55e}\n.status-banner.status-error{border-left:3px solid #ef4444}\n.status-pulse{width:8px;height:8px;border-radius:50%;background:#9ca3af;flex-shrink:0}\n.status-pulse.running{background:#6c63ff;animation:pulse 1.4s infinite}\n.status-pulse.done{background:#22c55e}\n.status-pulse.error{background:#ef4444}\n.status-text{font-size:14px;font-weight:500}\n.workflow-output-grid video{width:100%;border-radius:12px;margin-top:8px}\n\n/* ── Review grid ── */\n.review-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:20px}\n.review-item{background:#f9fafb;border-radius:10px;padding:12px 14px}\n.review-key{display:block;font-size:11px;color:#9ca3af;margin-bottom:4px;text-transform:uppercase;letter-spacing:.05em}\n.review-val{font-size:14px;font-weight:600;color:#1a1a2e;word-break:break-word}\n\n/* ── Chat ── */\n.chat-details{background:#f9fafb;border-radius:12px;margin-bottom:20px;border:1px solid #e8eaf0}\n.chat-details summary{padding:14px 16px;font-size:14px;font-weight:600;cursor:pointer;list-style:none}\n.chat-body{max-height:180px;overflow-y:auto;padding:10px 16px;display:flex;flex-direction:column;gap:8px}\n.chat-bubble{padding:9px 13px;border-radius:10px;font-size:13px;max-width:85%;line-height:1.5}\n.chat-bubble.bot{background:#ede9ff;color:#3730a3;align-self:flex-start;border-radius:10px 10px 10px 2px}\n.chat-bubble.user{background:#6c63ff;color:#fff;align-self:flex-end;border-radius:10px 10px 2px 10px}\n.chat-input-row{display:flex;gap:8px;padding:10px 16px;border-top:1px solid #e8eaf0}\n.chat-input-row input{flex:1}\n.chat-input-row button{background:#6c63ff;color:#fff;padding:9px 16px;border-radius:8px;flex-shrink:0}\n\n/* ── Submit ── */\n.submit-area{text-align:center;margin:8px 0 20px}\n.btn-submit{background:#6c63ff;color:#fff;padding:14px 40px;font-size:16px;font-weight:700;border-radius:12px;width:100%;max-width:320px}\n.btn-submit:hover{background:#574fd6;transform:translateY(-1px)}\n.hint{font-size:12px;color:#9ca3af;margin-top:8px}\n\n/* ── Error banner ── */\n.error-banner{background:#fef2f2;border:1px solid #fca5a5;border-radius:10px;padding:12px 16px;font-size:13px;color:#b91c1c;margin-bottom:16px}\n\n/* ── Misc ── */\n.intermediate-results img{max-width:100%;border-radius:8px}\n.thumb{width:48px;height:48px;object-fit:cover;border-radius:6px;border:2px solid #2d2d4a}\n.result-video-hero{background:#000;border-radius:16px;overflow:hidden;margin-bottom:20px}\n.result-video-hero video{width:100%;max-height:420px;display:block}\n.result-meta-row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:20px}\n.result-meta-item{background:#fff;border-radius:10px;padding:12px 16px;flex:1;min-width:120px;border:1px solid #e8eaf0}\n.result-meta-label{font-size:11px;color:#9ca3af;margin-bottom:4px}\n.result-meta-val{font-size:14px;font-weight:700;color:#1a1a2e}\n.result-section{background:#fff;border-radius:12px;padding:20px;margin-bottom:12px;border:1px solid #e8eaf0}\n.result-section h3{font-size:14px;font-weight:700;margin-bottom:12px;color:#374151}\n.result-section table{width:100%;border-collapse:collapse;font-size:13px}\n.result-section th{text-align:left;padding:8px 10px;background:#f9fafb;font-size:12px;color:#6b7280;border-bottom:1px solid #e8eaf0}\n.result-section td{padding:8px 10px;border-bottom:1px solid #f3f4f6;vertical-align:top}\n.result-section ul{padding-left:18px;line-height:1.8;font-size:13px;color:#374151}\n.result-section li{margin-bottom:4px}\n.result-collapse{margin-bottom:12px}\n.result-collapse summary{display:flex;align-items:center;gap:8px;background:#fff;border:1px solid #e8eaf0;border-radius:12px;padding:14px 18px;cursor:pointer;font-size:14px;font-weight:600;color:#374151;list-style:none}\n.result-collapse[open] summary{border-bottom-left-radius:0;border-bottom-right-radius:0;border-bottom-color:#f3f4f6}\n.result-collapse-body{background:#fff;border:1px solid #e8eaf0;border-top:none;border-radius:0 0 12px 12px;padding:16px 18px}\n.factor-tag{display:inline-block;background:#ede9ff;color:#6c63ff;border-radius:10px;padding:3px 10px;font-size:12px;margin:2px 3px}\n</style>\n</head>\n<body>\n%%BODY%%\n<script>\n// ── Step navigation ──\nfunction goStep(n){\n  document.querySelectorAll(\'.step-card\').forEach(c=>c.classList.remove(\'active\'));\n  const card=document.getElementById(\'step-\'+n);\n  if(card){card.classList.add(\'active\');card.scrollIntoView({behavior:\'smooth\',block:\'start\'});}\n  document.querySelectorAll(\'.si-dot\').forEach(d=>{\n    const s=parseInt(d.dataset.step);\n    d.classList.toggle(\'active\',s===n);\n    d.classList.toggle(\'done\',s<n);\n  });\n  if(n===4) buildReview();\n}\n\n// ── Chip selection ──\nfunction selectChip(btn,hiddenId){\n  const grp=btn.closest(\'.chip-group\');\n  grp.querySelectorAll(\'.chip\').forEach(c=>c.classList.remove(\'selected\'));\n  btn.classList.add(\'selected\');\n  const h=document.getElementById(hiddenId);\n  if(h) h.value=btn.dataset.val;\n}\n// toggleChip: 多选模式，逗号分隔\nfunction toggleChip(btn,hiddenId){\n  btn.classList.toggle(\'selected\');\n  const h=document.getElementById(hiddenId);\n  if(!h) return;\n  const grp=btn.closest(\'.chip-group\');\n  const vals=Array.from(grp.querySelectorAll(\'.chip.selected\')).map(c=>c.dataset.val).filter(Boolean);\n  h.value=vals.join(\',\');\n}\n// restore chips: hidden input lives INSIDE chip-group as first child\nfunction restoreChips(){\n  const multiIds=new Set([\'f-scene\',\'f-audience\']);\n  document.querySelectorAll(\'.chip-group\').forEach(grp=>{\n    const hidden=grp.querySelector(\'input[type="hidden"]\');\n    if(!hidden||!hidden.value) return;\n    const isMulti=multiIds.has(hidden.id);\n    const vals=isMulti?hidden.value.split(\',\').map(v=>v.trim()).filter(Boolean):[hidden.value];\n    grp.querySelectorAll(\'.chip\').forEach(c=>{\n      if(vals.includes(c.dataset.val)) c.classList.add(\'selected\');\n    });\n  });\n}\n// ── Style card selection ──\nfunction selectStyle(card){\n  document.querySelectorAll(\'.style-card\').forEach(c=>c.classList.remove(\'selected\'));\n  card.classList.add(\'selected\');\n  const h=document.getElementById(\'f-style\');\n  if(h) h.value=card.dataset.val;\n}\nfunction restoreStyle(){\n  const h=document.getElementById(\'f-style\');\n  if(!h||!h.value) return;\n  document.querySelectorAll(\'.style-card\').forEach(c=>{\n    if(c.dataset.val===h.value) c.classList.add(\'selected\');\n  });\n}\n\n// ── File upload preview ──\nfunction handleFiles(input){\n  const grid=document.getElementById(\'preview-grid\');\n  grid.innerHTML=\'\';\n  Array.from(input.files).forEach(f=>{\n    const img=document.createElement(\'img\');\n    img.className=\'preview-thumb\';\n    img.src=URL.createObjectURL(f);\n    grid.appendChild(img);\n  });\n}\n// drag-and-drop\n(function(){\n  const zone=document.getElementById(\'upload-zone\');\n  if(!zone) return;\n  zone.addEventListener(\'dragover\',e=>{e.preventDefault();zone.classList.add(\'drag\');});\n  zone.addEventListener(\'dragleave\',()=>zone.classList.remove(\'drag\'));\n  zone.addEventListener(\'drop\',e=>{\n    e.preventDefault();zone.classList.remove(\'drag\');\n    const inp=document.getElementById(\'file-input\');\n    inp.files=e.dataTransfer.files;handleFiles(inp);\n  });\n})();\n\n// ── Selling points ──\n// addPreset: click a preset tag to add it as a selling point input row\nfunction addPreset(btn){\n  if(btn.classList.contains(\'used\')) return;\n  const val=btn.textContent.trim();\n  if(getSPCount()>=5){alert(\'最多添加 5 条卖点\');return;}\n  addSPRow(val);\n  btn.classList.add(\'used\');\n  syncSP();\n}\n// addSPInput: add a blank custom input row\nfunction addSPInput(){\n  if(getSPCount()>=5){alert(\'最多添加 5 条卖点\');return;}\n  addSPRow(\'\');\n}\nfunction getSPCount(){\n  return document.querySelectorAll(\'#sp-list .sp-input-item\').length;\n}\nfunction addSPRow(val){\n  const list=document.getElementById(\'sp-list\');\n  const item=document.createElement(\'div\');\n  item.className=\'sp-input-item\';\n  const inp=document.createElement(\'input\');\n  inp.type=\'text\';inp.placeholder=\'填写卖点，例如：超薄设计，轻松随身携带\';\n  inp.value=val;inp.maxLength=40;\n  inp.addEventListener(\'input\',syncSP);\n  const del=document.createElement(\'button\');\n  del.type=\'button\';del.textContent=\'删除\';\n  del.addEventListener(\'click\',function(){\n    // un-mark preset if it matches\n    document.querySelectorAll(\'.preset-tag\').forEach(p=>{\n      if(p.textContent.trim()===inp.value.trim()) p.classList.remove(\'used\');\n    });\n    item.remove();syncSP();\n  });\n  item.appendChild(inp);item.appendChild(del);\n  list.appendChild(item);\n  inp.focus();\n}\nfunction syncSP(){\n  const vals=Array.from(document.querySelectorAll(\'#sp-list .sp-input-item input\'))\n    .map(i=>i.value.trim()).filter(Boolean);\n  const h=document.getElementById(\'sp-hidden\');\n  if(h) h.value=vals.join(\',\');\n}\n// restore from server-side value (edit/re-submit flow)\n(function(){\n  const h=document.getElementById(\'sp-hidden\');\n  if(!h||!h.value) return;\n  const vals=h.value.split(\',\').map(v=>v.trim()).filter(Boolean);\n  vals.forEach(v=>{\n    addSPRow(v);\n    // mark preset as used if matches\n    document.querySelectorAll(\'.preset-tag\').forEach(p=>{\n      if(p.textContent.trim()===v) p.classList.add(\'used\');\n    });\n  });\n})();\n\n// ── Review grid ──\nfunction buildReview(){\n  const fields=[\n    [\'商品名称\',\'f-title\'],[\'品牌\',\'f-brand\'],[\'型号\',\'f-model\'],\n    [\'目标受众\',\'f-audience\'],[\'创作方向\',\'f-direction\'],\n  ];\n  const g=document.getElementById(\'review-grid\');if(!g) return;\n  g.innerHTML=\'\';\n  fields.forEach(([label,id])=>{\n    const el=document.getElementById(id);const val=el?el.value:\'\';\n    if(!val) return;\n    const d=document.createElement(\'div\');d.className=\'review-item\';\n    d.innerHTML=\'<span class="review-key">\'+label+\'</span><span class="review-val">\'+val+\'</span>\';\n    g.appendChild(d);\n  });\n  const sp=document.getElementById(\'sp-hidden\');\n  if(sp&&sp.value){\n    const d=document.createElement(\'div\');d.className=\'review-item\';\n    d.innerHTML=\'<span class="review-key">核心卖点</span><span class="review-val">\'+sp.value+\'</span>\';\n    g.appendChild(d);\n  }\n}\n\n// ── Chat ──\nfunction sendChat(){\n  const inp=document.getElementById(\'chat-input\');const msg=inp.value.trim();if(!msg) return;\n  const body=document.getElementById(\'chat-body\');\n  const user=document.createElement(\'div\');user.className=\'chat-bubble user\';user.textContent=msg;body.appendChild(user);\n  inp.value=\'\';\n  const hist=document.getElementById(\'chat-history\');\n  if(hist) hist.value=(hist.value?hist.value+\'\\n\':\'\')+msg;\n  const bot=document.createElement(\'div\');bot.className=\'chat-bubble bot\';bot.textContent=\'已记录，生成时会纳入参考。\';body.appendChild(bot);\n  body.scrollTop=body.scrollHeight;\n}\n\n// ── Submit guard ──\nfunction checkSufficiencyAndSubmit(e){\n  syncSP();\n  const title=document.getElementById(\'f-title\');\n  if(!title||!title.value.trim()){e.preventDefault();alert(\'请填写商品名称\');goStep(2);return false;}\n  const sp=document.getElementById(\'sp-hidden\');\n  if(!sp||!sp.value.trim()){e.preventDefault();alert(\'请至少添加一个卖点\');goStep(2);return false;}\n  const btn=document.getElementById(\'btn-submit\');\n  if(btn){btn.disabled=true;btn.innerHTML=\'<span class="submit-icon">...</span> 正在提交，请稍候\';}\n  return true;\n}\n\n// ── Result page polling ──\nconst TASK_ID = "%%TASK_ID%%";\nconst IS_PROCESSING = %%IS_PROCESSING%%;\n\nif(TASK_ID && IS_PROCESSING){\n  setInterval(function(){ location.reload(); }, 5000);\n}\n\nrestoreChips();\nrestoreStyle();\n</script>\n</body>\n</html>\n'
def _html_response(content: str) -> HTMLResponse:
    """统一返回禁止缓存的 HTML 响应。"""

    return HTMLResponse(
        content=content,
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


def _render_generation_progress_card(success_task: dict) -> str:
    """渲染生成剧本/视频时的专用进度页，不混入审阅和结果内容。"""

    progress = int(success_task.get("workflow_progress", 0) or 0)
    stage = str(success_task.get("workflow_stage", "planning"))
    stage_label = {
        "preflight": "素材预检",
        "planning": "生成剧本",
        "asset_analysis": "理解素材",
        "script_plan": "生成剧本",
        "director_storyboard": "生成分镜",
        "shootability_review": "检查可拍性",
        "asset_matching": "匹配素材",
        "creation_plan": "准备视频生成",
        "render_video": "生成视频",
        "ab_variant": "生成候选方案",
        "content_review": "检查成片",
        "final_check": "整理结果",
    }.get(stage, stage)
    events_html = _render_workflow_events(success_task.get("workflow_events", []))
    stage_steps = _render_stage_checklist(stage)
    return f"""
    <section class="generation-progress-panel">
      <div class="generation-progress-head">
        <div>
          <h3>{escape(stage_label)}</h3>
          <p>{escape(str(success_task.get("workflow_message", "正在处理，请稍候。")))}</p>
          <p class="progress-note">这是长耗时任务，可关闭页面后继续查看；任务状态会自动刷新。</p>
        </div>
        <strong>{progress}%</strong>
      </div>
      <div class="progress-bar"><span style="width:{max(0, min(100, progress))}%"></span></div>
      <div class="stage-checklist">{stage_steps}</div>
      <div class="result-section compact">
        <h3>最近进度</h3>
        <ul class="event-list">{events_html}</ul>
      </div>
    </section>
    """


def _render_stage_checklist(current_stage: str) -> str:
    """渲染面向用户的短阶段清单，避免长任务期间只看到百分比。"""

    stage_order = [
        ("asset_analysis", "理解素材", "识别商品外观和素材用途"),
        ("script_plan", "生成剧本", "组织卖点、场景和节奏"),
        ("director_storyboard", "拆分镜", "准备每个镜头怎么拍"),
        ("render_video", "生成视频", "调用视频模型并拼接"),
        ("content_review", "检查成片", "检查商品和画面是否合理"),
        ("final_check", "整理结果", "输出预览和可复核记录"),
    ]
    aliases = {
        "preflight": "asset_analysis",
        "requirement_structurization": "asset_analysis",
        "product_context": "asset_analysis",
        "shootability_review": "director_storyboard",
        "asset_matching": "director_storyboard",
        "asset_gap_completion": "director_storyboard",
        "creation_plan": "director_storyboard",
        "ab_variant": "render_video",
        "draft_ready": "final_check",
        "draft_needs_review": "final_check",
    }
    normalized_stage = aliases.get(current_stage, current_stage)
    current_index = next(
        (index for index, (stage, _, _) in enumerate(stage_order) if stage == normalized_stage),
        0,
    )
    items = []
    for index, (_stage, title, desc) in enumerate(stage_order):
        state = "done" if index < current_index else "active" if index == current_index else ""
        items.append(
            f"""
            <div class="stage-check-item {state}">
              <strong>{escape(title)}</strong>
              <span>{escape(desc)}</span>
            </div>
            """
        )
    return "".join(items)


def _render_primary_product_confirmation(success_task: dict) -> str:
    """渲染主商品候选框；只有存在歧义时才要求用户额外操作。"""

    task_id = str(success_task.get("task_id", ""))
    asset_blocks: list[str] = []
    for asset_index, asset in enumerate(success_task.get("uploaded_assets", [])):
        profile = asset.get("primary_product", {})
        if not profile.get("requires_user_confirmation"):
            continue
        candidates = profile.get("candidates", [])
        source_size = profile.get("source_size", [])
        if not isinstance(candidates, list) or len(source_size) != 2:
            continue
        width, height = source_size
        if not width or not height:
            continue

        boxes: list[str] = []
        options: list[str] = []
        for candidate_index, candidate in enumerate(candidates):
            bbox = candidate.get("bbox", [])
            if len(bbox) != 4:
                continue
            left, top, right, bottom = bbox
            box_style = (
                f"left:{left / width * 100:.2f}%;top:{top / height * 100:.2f}%;"
                f"width:{(right - left) / width * 100:.2f}%;height:{(bottom - top) / height * 100:.2f}%;"
            )
            boxes.append(
                f'<div class="candidate-box" style="{box_style}"><span>{candidate_index + 1}</span></div>'
            )
            score = float(candidate.get("score", 0) or 0)
            options.append(
                '<label class="candidate-option">'
                f'<input id="selection_{asset_index}_{candidate_index}" type="radio" '
                f'name="selections" value="{asset_index}:{candidate_index}" '
                f'{"checked" if candidate_index == 0 else ""} required>'
                f'<span>候选 {candidate_index + 1} <small>自动评分 {score:.2f}</small></span>'
                '</label>'
            )

        asset_blocks.append(
            '<div class="candidate-asset">'
            f'<div class="candidate-image"><img src="{escape(str(asset.get("public_url", "")))}" '
            f'alt="{escape(str(asset.get("filename", "上传素材")))}">{"".join(boxes)}</div>'
            f'<div class="candidate-options"><strong>{escape(str(asset.get("filename", "上传素材")))}</strong>'
            f'{"".join(options)}</div>'
            '</div>'
        )

    if not asset_blocks:
        return ""
    return (
        '<section class="primary-product-confirmation">'
        '<h3>请确认本次视频需要推广的主商品</h3>'
        '<p>检测到画面中存在多个相近商品。请选择需要作为外观锚点的商品，确认后系统才会开始生成视频。</p>'
        f'<form method="post" action="/tasks/{escape(task_id)}/primary-product-confirmation">'
        f'{"".join(asset_blocks)}'
        '<div class="actions"><button type="submit">确认主商品并开始生成</button></div>'
        '</form></section>'
    )


def _render_script_review_panel(success_task: dict) -> str:
    """渲染可读、可编辑的剧本确认面板。"""

    if success_task.get("workflow_stage") != "script_review":
        return ""
    workflow_result = success_task.get("workflow_result", {}) or {}
    variants = _script_review_variant_map(workflow_result)
    if not variants:
        return ""

    task_id = escape(str(success_task.get("task_id", "")))
    variant_cards: list[str] = []
    for variant_index, (variant_id, variant) in enumerate(variants.items(), start=1):
        script_plan = variant.get("script_plan", {}) or {}
        storyboard = list(variant.get("storyboard") or [])
        readable_script = variant.get("readable_script", {}) or {}
        label, description = _public_script_variant_meta(variant_id)
        label = str(variant.get("label") or label)
        description = str(variant.get("description") or description)
        field_prefix = f"variant_{_safe_script_variant_key(variant_id)}__"
        synopsis = (
            str(script_plan.get("rich_story_text") or readable_script.get("synopsis") or "").strip()
            or _compose_script_synopsis_for_page(storyboard)
        )
        body_text = _script_variant_body_text(script_plan)
        timeline_html = _render_script_variant_timeline(storyboard)
        shot_forms = []
        for index, shot in enumerate(storyboard):
            shot_forms.append(
                f"""
                <div class="script-shot-card">
                  <div class="script-shot-head">
                    <strong>分镜 {escape(str(shot.get('shot_index', index + 1)))}</strong>
                    <label>时长
                      <input name="{field_prefix}shot_{index}_duration" type="number" min="1" max="15" value="{escape(str(shot.get('duration_seconds', 3)))}">
                    </label>
                  </div>
                  <label>这段要表达什么
                    <textarea name="{field_prefix}shot_{index}_scene_goal" rows="2">{escape(_shot_public_goal(shot))}</textarea>
                  </label>
                  <label>画面怎么拍
                    <textarea name="{field_prefix}shot_{index}_action" rows="3">{escape(_shot_public_action(shot))}</textarea>
                  </label>
                  <label>字幕 / 旁白
                    <input name="{field_prefix}shot_{index}_subtitle" value="{escape(str(shot.get('subtitle', '')))}">
                  </label>
                </div>
                """
            )
        shot_count = len(storyboard)
        variant_cards.append(
            f"""
            <article class="script-variant-card">
              <div class="script-variant-choice">
                <span>
                  <strong>{escape(label)}</strong>
                  <em>{escape(description)}</em>
                </span>
              </div>
              <div class="script-main-card">
                <label>总剧本
                  <textarea name="{field_prefix}script_synopsis" rows="5">{escape(_clean_public_script_text(synopsis, max_chars=600))}</textarea>
                </label>
                {timeline_html}
                <div class="script-line-grid">
                  <label>开场
                    <textarea name="{field_prefix}script_hook" rows="2">{escape(_clean_public_script_text(script_plan.get('hook', ''), max_chars=220))}</textarea>
                  </label>
                  <label>结尾引导
                    <textarea name="{field_prefix}script_cta" rows="2">{escape(_clean_public_script_text(script_plan.get('cta', ''), max_chars=220))}</textarea>
                  </label>
                </div>
                <label>卖点展开
                  <textarea name="{field_prefix}script_body" rows="3">{escape(_clean_public_script_text(body_text, max_chars=420))}</textarea>
                </label>
              </div>
              <input type="hidden" name="{field_prefix}shot_count" value="{shot_count}">
              <h4 class="script-section-title">分镜内容</h4>
              <div class="script-shot-grid compact">{"".join(shot_forms)}</div>
            </article>
            """
        )

    return f"""
    <section class="script-review-panel focused-script-page">
      <div class="script-review-head">
        <div>
          <h3>确认剧本方案</h3>
          <p>下面同时给出两个方向。你可以分别编辑每一版的文案和分镜，通过后系统会分别生成两条视频用于对比。</p>
        </div>
        <span>待确认</span>
      </div>
      <div class="script-review-action-bar">
        <div>
          <strong>剧本已生成，等待你确认</strong>
          <span>可以直接通过继续生成视频，也可以编辑下方任意一版后再通过。</span>
        </div>
        <div class="script-review-action-buttons">
          <button type="submit" form="script-review-form-{task_id}" class="btn-submit">通过并继续生成视频</button>
          <button
            type="submit"
            form="script-review-form-{task_id}"
            class="btn-back"
            name="feedback"
            value=""
            formaction="/tasks/{task_id}/script-review/regenerate"
            onclick="this.value=(document.querySelector('#script-review-form-{task_id} [name=reviewer_note]')||{{value:''}}).value"
          >重新生成剧本</button>
        </div>
      </div>
      <form id="script-review-form-{task_id}" method="post" action="/tasks/{task_id}/script-review/approve" class="script-review-form">
        <div class="script-variant-grid">{"".join(variant_cards)}</div>
        <label>给后续生成的备注 / 重新生成意见（可选）
          <textarea name="reviewer_note" rows="2" placeholder="例如：第二个镜头更强调通勤痛点，字幕更短一些。"></textarea>
        </label>
        <div class="script-review-actions">
          <button type="submit" class="btn-submit">通过并继续生成视频</button>
          <button
            type="submit"
            class="btn-back"
            name="feedback"
            value=""
            formaction="/tasks/{task_id}/script-review/regenerate"
            onclick="this.value=(this.form.querySelector('[name=reviewer_note]')||{{value:''}}).value"
          >重新生成剧本</button>
        </div>
      </form>
    </section>
    """


def _compose_script_synopsis_for_page(storyboard: list[dict]) -> str:
    goals = [
        str(shot.get("scene_goal") or shot.get("purpose") or "").strip()
        for shot in storyboard
        if str(shot.get("scene_goal") or shot.get("purpose") or "").strip()
    ]
    if goals:
        return "这条视频先" + "，再".join(goals[:4]) + "，最后完成商品卖点表达和购买引导。"
    actions = [
        str(shot.get("action") or "").strip()
        for shot in storyboard
        if str(shot.get("action") or "").strip()
    ]
    if actions:
        return "这条视频通过" + "，".join(actions[:4]) + "来展示商品使用过程和结果。"
    return "这条视频围绕商品真实外观、核心卖点和使用场景，生成一条完整带货短视频。"


def _render_workflow_result(workflow_result: dict) -> str:
    """渲染 agent 工作流输出。"""

    if not workflow_result:
        return """
        <div class="result-section">
          <h3>工作流输出</h3>
          <p class="hint">当前任务还没有生成工作流输出。</p>
        </div>
        """

    asset_analysis = workflow_result.get("asset_analysis", {})
    product_context = workflow_result.get("product_context", {})
    script_plan = workflow_result.get("script_plan", {})
    storyboard = workflow_result.get("storyboard", [])
    creation_plan = workflow_result.get("creation_plan", {})
    asset_matching = workflow_result.get("asset_matching", [])
    asset_gap_completion = workflow_result.get("asset_gap_completion", {})
    render_result = workflow_result.get("render_result", {})
    content_review = workflow_result.get("content_review", {})
    ab_variants = workflow_result.get("ab_variants", {}) or {}
    final_check = workflow_result.get("final_check", {})
    script_review = workflow_result.get("script_review", {})
    storyboard_review = workflow_result.get("storyboard_review", {})
    review_attempts = workflow_result.get("review_attempts", [])
    director_decision = workflow_result.get("director_decision", {})
    trace_summary = workflow_result.get("trace_summary", {})
    product_identity_card = product_context.get("product_identity_card") or asset_analysis.get("product_identity_card", {})

    asset_items = "".join(
        f"<li>{escape(asset.get('filename', 'unknown'))} "
        f"<span>{escape(asset.get('asset_type', 'unknown'))} / {escape(asset.get('suggested_role', ''))}</span></li>"
        for asset in asset_analysis.get("assets", [])
    ) or "<li>没有可用素材记录</li>"

    script_body_items = "".join(
        f"<li>{escape(line)}</li>" for line in script_plan.get("body", [])
    ) or "<li>暂无卖点展开</li>"

    storyboard_rows = "".join(
        f"""
        <tr>
          <td>{escape(str(shot.get('shot_index', '')))}</td>
          <td>{escape(str(shot.get('duration_seconds', '')))} 秒</td>
          <td>{escape(shot.get('narrative_role') or shot.get('purpose', ''))}</td>
          <td>{escape(shot.get('scene_goal') or shot.get('purpose', ''))}</td>
          <td>
            <strong>开始：</strong>{escape(shot.get('initial_state') or shot.get('visual_description', ''))}<br>
            <strong>动作：</strong>{escape(shot.get('action', ''))}<br>
            <strong>结束：</strong>{escape(shot.get('final_state', ''))}
          </td>
          <td>{escape(shot.get('subtitle', ''))}</td>
        </tr>
        """
        for shot in storyboard
    ) or """
        <tr>
          <td colspan="5">暂无分镜</td>
        </tr>
    """

    issue_items = "".join(
        f"<li>{escape(issue)}</li>" for issue in final_check.get("issues", [])
    ) or "<li>未发现阻塞问题</li>"

    video_block = ""
    if render_result.get("success") and render_result.get("video_url"):
        video_block = f"""
        <video controls preload="metadata" src="{escape(render_result['video_url'])}"></video>
        <p class="hint">视频文件：{escape(render_result.get('video_path', ''))}</p>
        <p class="hint">生成引擎：{escape(render_result.get('render_mode', ''))}</p>
        """
    else:
        video_block = f"""
        <p class="hint">预览视频未生成：{escape(str(render_result.get('error') or '暂无结果'))}</p>
        """

    fallback_block = ""
    if render_result.get("fallback_from"):
        fallback_block = f"""
        <p class="hint">视频模型回退原因：{escape(str(render_result['fallback_from'].get('error')))}</p>
        """

    subtitle_overlay_block = ""
    if render_result.get("subtitle_overlay"):
        subtitle_overlay = render_result["subtitle_overlay"]
        subtitle_overlay_block = f"""
        <p class="hint">字幕叠加：{escape('成功' if subtitle_overlay.get('success') else '失败')} / {escape(str(subtitle_overlay.get('mode', 'unknown')))}</p>
        """
        if subtitle_overlay.get("error"):
            subtitle_overlay_block += f"""
            <p class="hint">字幕叠加错误：{escape(str(subtitle_overlay.get('error')))}</p>
            """

    render_strategy_items = "".join(
        f"<li>镜头 {escape(str(item.get('shot_index', '')))}："
        f"{escape(item.get('strategy', ''))}，{escape(item.get('note', ''))}</li>"
        for item in asset_matching
    ) or "<li>暂无渲染策略</li>"
    asset_gap_items = _render_asset_gap_completion(asset_gap_completion)

    if isinstance(review_attempts, int):
        review_attempt_items = (
            f"<li>脚本 + 分镜共审阅 {review_attempts} 次</li>"
            if review_attempts
            else "<li>暂无打回记录</li>"
        )
    else:
        review_attempt_items = "".join(
            f"<li>{escape(attempt.get('step', ''))} 第 {escape(str(attempt.get('attempt', '')))} 次："
            f"{escape('通过' if attempt.get('passed') else '未通过')}，"
            f"动作={escape(attempt.get('action', ''))}"
            f"{_render_attempt_issues(attempt.get('issues', []))}</li>"
            for attempt in review_attempts
        ) or "<li>暂无打回记录</li>"
    director_variant_rows = _render_director_variants(director_decision.get("candidate_variants", []))
    director_asset_advice = "".join(
        f"<li>{escape(str(item))}</li>" for item in director_decision.get("asset_advice", [])
    ) or "<li>暂无额外素材建议</li>"
    trace_summary_items = _render_trace_summary(trace_summary)
    seedance_trace_items = _render_seedance_trace(render_result)
    identity_card_block = _render_product_identity_card(product_identity_card)
    content_review_items = _render_content_review(content_review)

    factor_combo = director_decision.get("factor_combination", {}) or {}
    factor_tags_html = ""
    if factor_combo:
        dim_labels = {
            "narrative_framework": "叙事框架", "hook": "开场", "pacing": "节奏",
            "camera": "运镜", "visual_focus": "画面重心", "exit": "退场", "emotion": "情绪基调",
        }
        tags = []
        for key, label in dim_labels.items():
            val = factor_combo.get(key, "")
            if val:
                tags.append(f'<span class="factor-tag">{escape(label)}：{escape(str(val))}</span>')
        if tags:
            factor_tags_html = f'<div class="factor-tags">{"".join(tags)}</div>'
    director_decision_block = _render_director_decision_or_strategy_summary(
        director_decision,
        product_context,
        creation_plan,
        factor_tags_html,
        director_variant_rows,
        director_asset_advice,
    )

    # ── 视频对比 block ──
    video_hero_html = _render_video_comparison(render_result, ab_variants)

    # ── meta 行 ──
    render_mode_label = escape(render_result.get("render_mode", creation_plan.get("render_mode", "—")))
    subtitle_label = "有字幕" if render_result.get("subtitle_overlay", {}).get("success") else "无字幕"
    elapsed_label = escape(str(render_result.get("elapsed_seconds", "—")))
    meta_row_html = f'''<div class="result-meta-row">
      <div class="result-meta-item"><div class="result-meta-label">渲染引擎</div><div class="result-meta-val">{render_mode_label}</div></div>
      <div class="result-meta-item"><div class="result-meta-label">字幕</div><div class="result-meta-val">{subtitle_label}</div></div>
      <div class="result-meta-item"><div class="result-meta-label">审阅次数</div><div class="result-meta-val">{escape(str(review_attempts)) if isinstance(review_attempts, int) else escape(str(len(review_attempts)))}</div></div>
      <div class="result-meta-item"><div class="result-meta-label">耗时 (s)</div><div class="result-meta-val">{elapsed_label}</div></div>
    </div>'''

    return f"""
    {video_hero_html}
    {meta_row_html}
    {director_decision_block}
    <details class="result-collapse">
      <summary>商品识别卡</summary>
      <div class="result-collapse-body">
        {identity_card_block}
        <div class="result-section">
          <h3>素材分析</h3>
          <p>{escape(asset_analysis.get('semantic_summary', '暂无素材摘要'))}</p>
          <ul>{asset_items}</ul>
        </div>
        <div class="result-section">
          <h3>商品上下文</h3>
          <p><strong>{escape(product_context.get('product_title', ''))}</strong></p>
          <p>{escape(product_context.get('creative_goal', ''))}</p>
          <p class="hint">目标人群：{escape(product_context.get('audience', ''))}</p>
        </div>
      </div>
    </details>
    <details class="result-collapse">
      <summary>分镜详情</summary>
      <div class="result-collapse-body">
        <div class="result-section">
          <h3>剧本规划</h3>
          <p><strong>开场白：</strong>{escape(script_plan.get('hook', ''))}</p>
          <ul>{script_body_items}</ul>
          <p><strong>结尾引导：</strong>{escape(script_plan.get('cta', ''))}</p>
          <p class="hint">剧本审核：{escape('通过' if script_review.get('passed') else '需要确认')}</p>
        </div>
        <div class="result-section">
          <h3>分镜草稿</h3>
          <table>
            <thead><tr><th>镜头</th><th>时长</th><th>镜头功能</th><th>镜头目标</th><th>状态转移</th><th>字幕 / 旁白</th></tr></thead>
            <tbody>{storyboard_rows}</tbody>
          </table>
          <p class="hint">分镜审核：{escape('通过' if storyboard_review.get('passed') else '需要确认')}</p>
        </div>
        <div class="result-section">
          <h3>打回与重试记录</h3>
          <ul>{review_attempt_items}</ul>
        </div>
      </div>
    </details>
    <details class="result-collapse">
      <summary>素材匹配与缺口补全</summary>
      <div class="result-collapse-body">
        <div class="result-section">
          <h3>创作计划与检查</h3>
          <p>渲染模式：{escape(creation_plan.get('render_mode', ''))}</p>
          <p>画幅：{escape(creation_plan.get('aspect_ratio', ''))}</p>
          <p>总时长：{escape(str(creation_plan.get('total_duration_seconds', '')))} 秒</p>
          <ul>{render_strategy_items}</ul>
          <p class="hint">素材缺口补全</p>
          <ul>{asset_gap_items}</ul>
          <p>检查结果：{escape('通过' if final_check.get('passed') else '需要确认')}</p>
          <ul>{issue_items}</ul>
        </div>
      </div>
    </details>
    <details class="result-collapse">
      <summary>内容审核 / 生成 Trace</summary>
      <div class="result-collapse-body">
        <div class="result-section">
          <h3>内容审核与修复</h3>
          <ul>{content_review_items}</ul>
        </div>
        <div class="result-section">
          <h3>生成 Trace 摘要</h3>
          <ul>{trace_summary_items}</ul>
          <p class="hint">Seedance 分镜任务</p>
          <ul>{seedance_trace_items}</ul>
        </div>
        <div class="result-section">
          <h3>视频渲染详情</h3>
          {video_block}
          {fallback_block}
          {subtitle_overlay_block}
        </div>
      </div>
    </details>
    """


def _render_video_comparison(render_result: dict, ab_variants: dict) -> str:
    """把默认 A 和候选 B 同屏展示，便于用户直接比较。"""

    cards: list[str] = []
    a_url = _video_url_from_result(render_result)
    if render_result.get("success") and a_url:
        cards.append(
            f"""
            <div class="video-variant-card">
              <h3>方案 A：默认保真版</h3>
              <video controls preload="metadata" src="{escape(a_url)}"></video>
              {_render_video_export_actions(a_url, filename="product_video_a.mp4")}
              <ul>
                <li>优先保持上传商品外观和素材绑定稳定。</li>
                <li>适合检查商品是否跑偏、主体是否一致。</li>
              </ul>
            </div>
            """
        )

    for variant_id, variant in ab_variants.items():
        if not isinstance(variant, dict):
            continue
        variant_render = variant.get("render_result") or {}
        b_url = _video_url_from_result(variant_render) or _public_upload_url(str(variant.get("video_path", "")))
        if not variant.get("success") or not b_url:
            continue
        notes = "".join(
            f"<li>{escape(str(note))}</li>"
            for note in (variant.get("review_notes") or ["更偏场景化带货表达。"])[:3]
        )
        cards.append(
            f"""
            <div class="video-variant-card">
              <h3>方案 B：场景化带货版</h3>
              <video controls preload="metadata" src="{escape(b_url)}"></video>
              {_render_video_export_actions(b_url, filename="product_video_b.mp4")}
              <ul>{notes}</ul>
            </div>
            """
        )

    if not cards:
        return ""
    return f'<div class="video-comparison-grid">{"".join(cards)}</div>'


def _render_director_decision_or_strategy_summary(
    director_decision: dict,
    product_context: dict,
    creation_plan: dict,
    factor_tags_html: str,
    director_variant_rows: str,
    director_asset_advice: str,
) -> str:
    """有导演决策时展示决策；否则展示当前实际采用的创作策略摘要。"""

    has_director_decision = bool(
        director_decision.get("selected_strategy")
        or director_decision.get("candidate_variants")
        or director_decision.get("factor_combination")
    )
    if has_director_decision:
        return f"""
        <details class="result-collapse">
          <summary>AI 导演决策</summary>
          <div class="result-collapse-body">
            <div class="result-section">
              <p><strong>因子组合：</strong>{escape(director_decision.get('selected_strategy', '暂无策略'))}</p>
              {factor_tags_html}
              <p style="margin-top:8px">{escape(director_decision.get('selected_reason', '暂无选择理由'))}</p>
              <table>
                <thead><tr><th>候选版本</th><th>因子变动</th><th>开场角度</th><th>风格建议</th><th>预估分</th><th>风险</th></tr></thead>
                <tbody>{director_variant_rows}</tbody>
              </table>
              <p class="hint">素材建议</p>
              <ul>{director_asset_advice}</ul>
              <p class="hint">渲染建议：{escape(director_decision.get('render_advice', '暂无建议'))}</p>
            </div>
          </div>
        </details>
        """

    visual_style = product_context.get("visual_style_bible") or creation_plan.get("visual_style_bible") or {}
    style_summary = ""
    if isinstance(visual_style, dict):
        style_summary = str(visual_style.get("style_summary") or visual_style.get("user_style") or "").strip()
    strategy_items = [
        "当前任务使用素材保真 + A/B 剧本确认链路，不再单独输出旧版导演候选表。",
        f"视频风格：{style_summary or product_context.get('style') or '清晰直接的商品展示风格'}。",
        f"渲染方式：{creation_plan.get('render_mode') or '等待生成'}。",
        "商品镜头优先绑定上传素材；更大胆的带货场景通过 B 版候选在确认页单独呈现。",
    ]
    return f"""
    <details class="result-collapse">
      <summary>创作策略摘要</summary>
      <div class="result-collapse-body">
        <div class="result-section">
          <ul>{"".join(f"<li>{escape(item)}</li>" for item in strategy_items)}</ul>
        </div>
      </div>
    </details>
    """


def _render_video_export_actions(video_url: str, *, filename: str = "aigc_product_video.mp4") -> str:
    """渲染导出入口；只链接已有视频，不做额外转码以免影响生成链路。"""

    if not video_url:
        return ""
    escaped_url = escape(video_url)
    return f"""
    <div class="video-export-actions">
      <a href="{escaped_url}" download="{escape(filename)}">下载视频</a>
      <a href="{escaped_url}" target="_blank" rel="noopener">新窗口预览</a>
      <span>当前导出为系统生成的默认画幅；可用于演示和人工复核。</span>
    </div>
    """


def _video_url_from_result(result: dict) -> str:
    if not isinstance(result, dict):
        return ""
    path_url = _public_upload_url(str(result.get("video_path", "")))
    raw_url = str(result.get("video_url") or "")
    if path_url and "/variants/" in path_url:
        return path_url
    return raw_url or path_url


def _public_upload_url(video_path: str) -> str:
    normalized = video_path.replace("\\", "/")
    marker = ".uploads/"
    if marker not in normalized:
        return ""
    return "/uploads/" + normalized.split(marker, 1)[1]


def _render_asset_gap_completion(asset_gap_completion: dict) -> str:
    """渲染素材缺口补全记录，让用户知道系统是否真的处理了素材不足问题。"""

    if not asset_gap_completion:
        return "<li>暂无素材缺口补全记录</li>"

    records = asset_gap_completion.get("gap_records", [])
    if not records:
        note = asset_gap_completion.get("note", "当前分镜没有发现必须补全的素材缺口。")
        return f"<li>{escape(str(note))}</li>"

    return "".join(
        "<li>"
        f"镜头 {escape(str(record.get('shot_index', '')))}："
        f"{escape(str(record.get('original_strategy', '')))} → "
        f"{escape(str(record.get('final_strategy', '')))}，"
        f"{escape(str(record.get('note', '')))}"
        f"{_render_asset_gap_risk(record)}"
        "</li>"
        for record in records
    )


def _render_content_review(content_review: dict) -> str:
    """渲染内容级审核结果和局部修复建议。"""

    if not content_review:
        return "<li>暂无内容审核结果</li>"

    items = [
        f"修复前审核结果：{'通过' if content_review.get('passed') else '需要确认'}",
        f"审核模式：{content_review.get('mode', 'unknown')}",
        f"是否跳过：{content_review.get('skipped', False)}",
        f"结论：{content_review.get('summary', '')}",
    ]
    if content_review.get("error"):
        items.append(f"错误：{content_review.get('error')}")

    repair_execution = content_review.get("repair_execution") or {}
    if repair_execution:
        items.append(
            "实际修复："
            f"尝试 {repair_execution.get('attempted_count', 0)} 个，"
            f"成功 {repair_execution.get('succeeded_count', 0)} 个，"
            f"失败 {repair_execution.get('failed_count', 0)} 个，"
            f"跳过 {repair_execution.get('skipped_count', 0)} 个"
        )
        if repair_execution.get("reconcat_success"):
            items.append("修复后视频已重新合成。")
        for record in repair_execution.get("records", []):
            status_label = {
                "succeeded": "成功",
                "failed": "失败",
                "skipped": "跳过",
            }.get(record.get("status"), record.get("status", "未知"))
            items.append(
                "修复执行："
                f"镜头 {record.get('shot_index')}，"
                f"状态={status_label}，"
                f"策略={record.get('repair_strategy', '')}，"
                f"说明={record.get('error', record.get('clip_path', ''))}"
            )

    for repair in content_review.get("repair_records", []):
        items.append(
            "修复建议："
            f"镜头 {repair.get('shot_index')}，"
            f"动作={repair.get('action')}，"
            f"原因={repair.get('reason', '')}"
        )
    return "".join(f"<li>{escape(str(item))}</li>" for item in items)


def _render_product_identity_card(identity_card: dict) -> str:
    """渲染商品身份卡和动作能力，让用户能检查生成约束是否合理。"""

    if not identity_card:
        return """
        <div class="result-section">
          <h3>商品身份卡</h3>
          <p class="hint">暂无商品身份卡。</p>
        </div>
        """

    motion = identity_card.get("motion_affordance", {}) or {}
    must_preserve = "".join(
        f"<li>{escape(str(item))}</li>" for item in identity_card.get("must_preserve", [])
    ) or "<li>暂无必须保持项</li>"
    forbidden_changes = "".join(
        f"<li>{escape(str(item))}</li>" for item in identity_card.get("forbidden_changes", [])
    ) or "<li>暂无禁止变化项</li>"
    allowed_actions = "、".join(str(item) for item in motion.get("allowed_actions", [])) or "暂无"
    forbidden_actions = "、".join(str(item) for item in motion.get("forbidden_actions", [])) or "暂无"

    return f"""
    <div class="result-section">
      <h3>商品身份卡</h3>
      <p><strong>{escape(str(identity_card.get('product_type', '未知商品')))}</strong> / 主色：{escape(str(identity_card.get('primary_color', 'unknown')))}</p>
      <p>{escape(str(identity_card.get('appearance_summary', '暂无外观摘要')))}</p>
      <p class="hint">必须保持</p>
      <ul>{must_preserve}</ul>
      <p class="hint">禁止变化</p>
      <ul>{forbidden_changes}</ul>
      <p class="hint">允许动作：{escape(allowed_actions)}</p>
      <p class="hint">禁止动作：{escape(forbidden_actions)}</p>
    </div>
    """


def _render_asset_gap_risk(record: dict) -> str:
    """渲染素材缺口的真实性风险提示。"""

    risk = str(record.get("risk", "")).strip()
    if not risk:
        return ""
    return f"<span>风险：{escape(risk)}</span>"


def _render_attempt_issues(issues: list) -> str:
    """渲染某次重试失败的原因。"""

    if not issues:
        return ""
    issue_text = "；问题：" + "、".join(str(issue) for issue in issues)
    return escape(issue_text)


def _bool_js(value: bool) -> str:
    return "true" if value else "false"


def _render_workflow_events(events: list[dict]) -> str:
    """渲染后台工作流已经写入的进度事件。"""

    if not events:
        return "<li>暂无执行事件</li>"

    recent = events[-10:]
    return "".join(
        "<li class='event-item'>"
        f"<strong>{escape(str(e.get('progress', 0)))}% &middot; {escape(str(e.get('stage', 'unknown')))}</strong>"
        f"{_escape_event_msg(str(e.get('message', '')))}"
        f"<span>{escape(str(e.get('created_at', '')))}</span>"
        "</li>"
        for e in recent
    )


def _escape_event_msg(msg: str) -> str:
    if "重试" in msg or "retry" in msg:
        return f"<span class='stage-sub retrying'>{escape(msg)}</span>"
    if "失败" in msg or "超时" in msg or "错误" in msg:
        return f"<span class='stage-sub error'>{escape(msg)}</span>"
    return escape(msg)


def _render_director_variants(variants: list[dict]) -> str:
    """渲染 Director 生成的候选创意版本。"""

    if not variants:
        return """
        <tr>
          <td colspan="6">暂无候选版本</td>
        </tr>
        """

    return "".join(
        f"""
        <tr>
          <td>{escape(str(variant.get('name', '')))}</td>
          <td style="font-size:12px;color:var(--muted)">{escape(str(variant.get('factor_diff', '')))}</td>
          <td>{escape(str(variant.get('hook_angle', '')))}</td>
          <td>{escape(str(variant.get('style_notes', '')))}</td>
          <td>{escape(str(variant.get('estimated_score', '')))}</td>
          <td>{escape(str(variant.get('risk', '')))}</td>
        </tr>
        """
        for variant in variants
    )


def _render_trace_summary(trace_summary: dict) -> str:
    if not trace_summary:
        return "<li>暂无 Trace 摘要</li>"
    llm_usage = trace_summary.get("llm_usage", {})
    llm_errors = trace_summary.get("llm_errors", {})
    items = [
        f"AI 素材分析：{llm_usage.get('asset_analysis', False)}",
        f"AI 分镜规划：{llm_usage.get('director_storyboard', False)}",
        f"AI 剧本生成：{llm_usage.get('script_plan', False)}",
        f"审核轮次：{trace_summary.get('review_attempt_count', 0)}",
        f"重试次数：{trace_summary.get('retry_count', 0)}",
        f"生成引擎：{trace_summary.get('render_mode', 'unknown')}",
        f"使用备用方案：{trace_summary.get('fallback_used', False)}",
        f"素材不足数：{trace_summary.get('asset_gap_count', 0)}",
        f"未解决素材不足：{trace_summary.get('asset_gap_unresolved_count', 0)}",
        f"素材真实性风险：{trace_summary.get('asset_gap_risk_count', 0)}",
        f"AI 内容审核：{llm_usage.get('content_review', False)}",
        f"内容审核通过：{trace_summary.get('content_review_passed', True)}",
        f"内容审核跳过：{trace_summary.get('content_review_skipped', False)}",
        f"修复建议：{trace_summary.get('content_repair_count', 0)}",
        f"生成报告通过：{trace_summary.get('final_check_passed', True)}",
        f"其他提醒：{trace_summary.get('final_issue_count', 0)}",
    ]
    if llm_errors.get("asset_analysis"):
        items.append(f"素材理解错误：{llm_errors['asset_analysis']}")
    return "".join(f"<li>{escape(str(item))}</li>" for item in items)
def _render_seedance_trace(render_result: dict) -> str:
    """渲染 Seedance 子任务执行情况。"""

    shot_results = render_result.get("shot_results", [])
    fallback_from = render_result.get("fallback_from") or {}
    if not shot_results and fallback_from:
        shot_results = fallback_from.get("shot_results", [])

    if not shot_results:
        return "<li>暂无 Seedance 子任务记录</li>"

    return "".join(
        "<li>"
        f"镜头 {escape(str(item.get('shot_index', '')))}："
        f"success={escape(str(item.get('success', False)))}, "
        f"status={escape(str(item.get('status', 'unknown')))}, "
        f"task_id={escape(str(item.get('seedance_task_id', item.get('task_id', ''))))}, "
        f"elapsed={escape(str(item.get('elapsed_seconds', '')))}s"
        f"{_render_seedance_error(item)}"
        "</li>"
        for item in shot_results
    )


def _render_seedance_error(item: dict) -> str:
    """渲染 Seedance 子任务错误摘要。"""

    if not item.get("error"):
        return ""
    return f"，error={escape(str(item.get('error')))}"


def _render_asset_link(asset: dict) -> str:
    """渲染已保存素材的访问链接。"""

    public_url = asset.get("public_url")
    if not public_url:
        return ""
    return f' <a href="{escape(public_url)}" target="_blank">查看素材</a>'


def _render_stage_panel(success_task: dict, events: list[dict]) -> str:
    """根据任务当前阶段和事件流渲染工作流阶段面板。JS 轮询后会动态更新。"""

    current_stage = str(success_task.get("workflow_stage", "created"))
    event_by_stage: dict[str, dict] = {}
    for event in events:
        event_by_stage[str(event.get("stage", ""))] = event

    stage_order = [
        ("asset_analysis", "素材理解"),
        ("requirement_structurization", "需求整理"),
        ("product_context", "商品上下文"),
        ("script_plan", "剧本规划"),
        ("director_storyboard", "导演分镜"),
        ("narrative_review", "叙事审核"),
        ("asset_matching", "素材匹配"),
        ("asset_gap_completion", "缺口补全"),
        ("creation_plan", "创作计划"),
        ("render_video", "视频渲染"),
        ("content_review", "内容审核"),
        ("final_check", "生成报告"),
        ("draft_ready", "草稿就绪"),
    ]

    if success_task.get("status") in {"completed", "needs_review"}:
        completed = {stage for stage, _ in stage_order}
    else:
        completed = {str(e.get("stage", "")) for e in events}

    items = []
    current_seen = False
    for stage, label in stage_order:
        event = event_by_stage.get(stage)
        if stage in completed and stage != current_stage:
            state, state_text = "done", "已完成"
        elif stage == current_stage:
            state, state_text = "active", "进行中"
            current_seen = True
        elif current_seen:
            state, state_text = "idle", "未开始"
        else:
            state = "done" if stage in completed else "idle"
            state_text = "已完成" if stage in completed else "未开始"

        sub = str(event.get("message", "")) if event else ""
        sub_class = ""
        if "重试" in sub or "retry" in sub:
            sub_class = " retrying"
        elif "失败" in sub or "超时" in sub or "错误" in sub:
            sub_class = " error"

        sub_html = ""
        if sub:
            sub_html = '<span class="stage-sub{}">{}</span>'.format(sub_class, escape(sub))
        stage_html = (
            '<div class="stage-item {}">'.format(state)
            + '<span class="stage-dot"></span>'
            + '<div class="stage-info">'
            + '<span class="stage-label">{} <em>{}</em></span>'.format(escape(label), state_text)
            + sub_html
            + '</div></div>'
        )
        items.append(stage_html)

    if current_stage == "failed":
        items.append(
            '<div class="stage-item failed"><span class="stage-dot"></span>'
            '<div class="stage-info"><span class="stage-label">执行失败</span>'
            f'<span class="stage-sub error">{escape(str(success_task.get("workflow_message", "")))}</span>'
            '</div></div>'
        )

    return "".join(items)


def _stop_previous_project_server(host: str, port: int) -> None:
    """启动前清理同项目遗留监听进程，避免固定端口被旧后台服务占用。"""

    if os.getenv("AIGC_AUTO_STOP_STALE_SERVER", "1") == "0":
        return

    for pid in _listener_pids_on_port(port):
        if pid == os.getpid():
            continue
        if not _pid_looks_like_this_app(pid):
            _flow_print(
                "[task_creation_demo_app] 端口已被其他进程占用，未自动结束："
                f"host={host}, port={port}, pid={pid}"
            )
            continue

        _flow_print(
            "[task_creation_demo_app] 检测到旧服务仍占用端口，正在停止："
            f"host={host}, port={port}, old_pid={pid}"
        )
        _terminate_pid(pid)


def _listener_pids_on_port(port: int) -> set[int]:
    """返回当前监听指定 TCP 端口的进程 PID。仅依赖 Linux/WSL 常见的 ss 命令。"""

    try:
        result = subprocess.run(
            ["ss", "-ltnp"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return set()

    pids: set[int] = set()
    port_pattern = re.compile(rf":{re.escape(str(port))}(?=\s)")
    for line in result.stdout.splitlines():
        if not port_pattern.search(line):
            continue
        for match in re.finditer(r"pid=(\d+)", line):
            pids.add(int(match.group(1)))
    return pids


def _pid_looks_like_this_app(pid: int) -> bool:
    """只允许自动清理当前项目自己的 Python 服务，避免误杀其他占用 8010 的程序。"""

    proc_dir = Path("/proc") / str(pid)
    try:
        cmdline = (proc_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", errors="ignore")
    except OSError:
        return False

    if "task_creation_demo_app.py" not in cmdline and APP_FILE not in cmdline:
        return False

    try:
        cwd = os.path.realpath(os.readlink(proc_dir / "cwd"))
        app_dir = os.path.realpath(os.path.dirname(APP_FILE))
        if cwd == app_dir:
            return True
    except OSError:
        pass

    return os.path.realpath(APP_FILE) in cmdline


def _terminate_pid(pid: int) -> None:
    """先温和终止旧服务，短暂等待后再兜底强制结束。"""

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        _flow_print(f"[task_creation_demo_app] 无权限停止旧服务：pid={pid}")
        return

    for _ in range(30):
        if not _pid_is_alive(pid):
            return
        time.sleep(0.1)

    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        return


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8010"))
    host = os.getenv("HOST", "127.0.0.1")
    _stop_previous_project_server(host, port)
    print(
        f"[task_creation_demo_app] 准备启动任务创建演示服务：http://{host}:{port} "
        f"pid={SERVER_PID}, run_id={RUN_INSTANCE_ID}, started_at={SERVER_STARTED_AT}",
        flush=True,
    )
    logger.info("准备启动任务创建演示服务：http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port, access_log=False)
