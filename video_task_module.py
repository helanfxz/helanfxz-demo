"""
AIGC 电商视频系统的最小任务创建模块。

这个文件刻意不依赖任何 Web 框架，先把最基础的后端领域能力收住。
在接入 FastAPI、数据库、后台任务之前，我们先定义清楚下面五个对象：

1. 创建视频任务的输入对象
2. 任务状态模型
3. 任务实体
4. 一个简单的内存仓储
5. 一个负责校验并创建任务的函数

目标很明确：先让第一块代码可以被独立阅读、独立审查、独立扩展。

当前版本刻意控制抽象层级：
- 保留"输入对象"和"任务实体"，因为它们表达的是两种不同语义
- 保留"仓储"，因为后续大概率会替换成数据库实现
- 不额外保留只有单一职责的服务类，避免过早分层
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
import logging
import os
from typing import Any, Dict, List
from uuid import uuid4

logger = logging.getLogger("video_task_module")
VERBOSE_LOG = os.getenv("AIGC_VERBOSE_LOG") == "1"


def print(*args, **kwargs):  # type: ignore[override]
    """默认隐藏任务模块内部细节输出。"""

    if VERBOSE_LOG:
        builtins.print(*args, **kwargs)


def _flow_print(message: str) -> None:
    """输出任务模块关键状态。"""

    builtins.print(message, flush=True)


class TaskStatus(StrEnum):
    """
    第一版工作流使用的稳定任务状态。

    这里故意只保留少量状态，避免一开始把状态机做得过重。
    后续模块可以继续扩展，例如 `running`、`failed`、`succeeded`。
    """

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskValidationError(ValueError):
    """当任务输入不完整或违反基本规则时抛出。"""


@dataclass(slots=True)
class CreateVideoTaskCommand:
    """
    创建视频任务时需要的输入数据。

    这个命令对象是"外部输入"和"内部领域逻辑"之间的边界。
    将来接 API 时，应该先把请求 JSON 转成这个对象，再交给服务层处理。
    """

    title: str
    selling_points: List[str]
    target_platform: str
    duration_seconds: int
    style: str
    custom_style_prompt: str = ""
    uploaded_assets: List["UploadedAsset"] = field(default_factory=list)
    product_type: str = ""
    target_audience: str = ""
    usage_scene: str = ""
    creative_direction: str = ""
    forbidden_changes: List[str] = field(default_factory=list)
    chat_history: List[str] = field(default_factory=list)
    input_confidence: str = ""


@dataclass(slots=True)
class UploadedAsset:
    """
    前端上传后进入系统的最小素材元数据。

    当前阶段只记录最基本的信息，目的是让"上传 -> 创建任务"这条链可见。
    后续如果要做素材库，再继续补充哈希、尺寸、时长、切片结果等字段。
    """

    filename: str
    content_type: str
    file_path: str = ""
    public_url: str = ""
    file_size: int = 0
    asset_type: str = ""
    source_url: str = ""
    primary_product: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class VideoTask:
    """
    系统内部保存的任务实体。

    这里已经放入时间戳和稳定标识符，后续模块就可以继续往任务上挂：
    素材、剧本、分镜、渲染任务等信息。
    """

    task_id: str
    title: str
    selling_points: List[str]
    target_platform: str
    duration_seconds: int
    style: str
    custom_style_prompt: str
    product_type: str = ""
    target_audience: str = ""
    usage_scene: str = ""
    creative_direction: str = ""
    forbidden_changes: List[str] = field(default_factory=list)
    chat_history: List[str] = field(default_factory=list)
    input_confidence: str = ""
    structured_requirements: Dict[str, Any] = field(default_factory=dict)
    uploaded_assets: List[UploadedAsset] = field(default_factory=list)
    status: TaskStatus = TaskStatus.QUEUED
    workflow_stage: str = "created"
    workflow_message: str = ""
    workflow_progress: int = 0
    workflow_events: List[Dict[str, Any]] = field(default_factory=list)
    workflow_result: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        """
        把任务实体转换成适合 JSON 输出的字典。

        返回普通字典可以让这个模块继续保持独立，
        不需要提前绑定任何 Web 框架或序列化库。
        """

        return {
            "task_id": self.task_id,
            "title": self.title,
            "selling_points": self.selling_points,
            "target_platform": self.target_platform,
            "duration_seconds": self.duration_seconds,
            "style": self.style,
            "custom_style_prompt": self.custom_style_prompt,
            "product_type": self.product_type,
            "target_audience": self.target_audience,
            "usage_scene": self.usage_scene,
            "creative_direction": self.creative_direction,
            "forbidden_changes": self.forbidden_changes,
            "chat_history": self.chat_history,
            "input_confidence": self.input_confidence,
            "structured_requirements": self.structured_requirements,
            "uploaded_assets": [
                {
                    "filename": asset.filename,
                    "content_type": asset.content_type,
                    "file_path": asset.file_path,
                    "public_url": asset.public_url,
                    "file_size": asset.file_size,
                    "asset_type": asset.asset_type,
                    "source_url": asset.source_url,
                    "primary_product": asset.primary_product,
                }
                for asset in self.uploaded_assets
            ],
            "status": self.status.value,
            "workflow_stage": self.workflow_stage,
            "workflow_message": self.workflow_message,
            "workflow_progress": self.workflow_progress,
            "workflow_events": self.workflow_events,
            "workflow_result": self.workflow_result,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(slots=True)
class InMemoryTaskRepository:
    """
    早期开发阶段使用的临时内存仓储。

    这里刻意保持简单。
    只要你认可当前模块边界，后面就可以把它替换成数据库仓储，
    同时不改服务层接口。
    """

    _tasks: Dict[str, VideoTask] = field(default_factory=dict)

    def save(self, task: VideoTask) -> VideoTask:
        print(f"[video_task_module] 任务写入内存仓储：task_id={task.task_id}", flush=True)
        logger.info("任务写入内存仓储：task_id=%s", task.task_id)
        self._tasks[task.task_id] = task
        return task

    def get(self, task_id: str) -> VideoTask:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise KeyError(f"任务 '{task_id}' 不存在。") from exc

    def update(self, task: VideoTask) -> VideoTask:
        print(f"[video_task_module] 更新内存仓储中的任务：task_id={task.task_id}", flush=True)
        logger.info("更新内存仓储中的任务：task_id=%s", task.task_id)
        self._tasks[task.task_id] = task
        return task


def create_video_task(
    command: CreateVideoTaskCommand,
    repository: InMemoryTaskRepository,
) -> VideoTask:
    """
    创建一个新的带货视频任务。

    这里使用模块级函数，而不是额外包一层服务类。
    原因是当前逻辑很薄，只有"校验 -> 创建实体 -> 保存"三步，
    用函数表达会更直接，也更容易在后续接 API 时复用。
    """

    print("[video_task_module] 开始执行任务创建逻辑。", flush=True)
    logger.info("开始执行任务创建逻辑。")
    _validate_create_command(command)
    print("[video_task_module] 输入校验通过。", flush=True)
    logger.info("输入校验通过。")

    current_time = _utc_now()
    task = VideoTask(
        task_id=_generate_task_id(),
        title=command.title.strip(),
        selling_points=[point.strip() for point in command.selling_points if point.strip()],
        target_platform=command.target_platform.strip().lower(),
        duration_seconds=command.duration_seconds,
        style=command.style.strip(),
        custom_style_prompt=command.custom_style_prompt.strip(),
        product_type=command.product_type.strip(),
        target_audience=command.target_audience.strip(),
        usage_scene=command.usage_scene.strip(),
        creative_direction=command.creative_direction.strip(),
        forbidden_changes=[item.strip() for item in command.forbidden_changes if item.strip()],
        chat_history=[item.strip() for item in command.chat_history if item.strip()],
        input_confidence=command.input_confidence.strip() or "medium",
        structured_requirements={},
        uploaded_assets=command.uploaded_assets,
        status=TaskStatus.QUEUED,
        workflow_stage="created",
        workflow_message="任务已创建，等待启动工作流。",
        workflow_progress=0,
        workflow_events=[
            _workflow_event("created", "任务已创建，等待启动工作流。", 0),
        ],
        workflow_result={},
        created_at=current_time,
        updated_at=current_time,
    )
    _flow_print(f"[video_task_module] 任务已创建：task_id={task.task_id}")
    logger.info("任务实体构建完成：task_id=%s", task.task_id)
    return repository.save(task)


def update_task_assets(
    task_id: str,
    repository: InMemoryTaskRepository,
    uploaded_assets: List[UploadedAsset],
) -> VideoTask:
    """保存文件落盘后的素材信息。"""

    print(f"[video_task_module] 准备更新任务素材：task_id={task_id}", flush=True)
    task = repository.get(task_id)
    task.uploaded_assets = uploaded_assets
    task.updated_at = _utc_now()
    return repository.update(task)


def update_task_primary_product_preflight(
    task_id: str,
    repository: InMemoryTaskRepository,
    profiles_by_path: Dict[str, Dict[str, Any]],
) -> VideoTask:
    """保存主商品预检结果；存在歧义时暂停在用户确认阶段。"""

    task = repository.get(task_id)
    confirmation_required = False
    for asset in task.uploaded_assets:
        profile = dict(profiles_by_path.get(asset.file_path, {}))
        if not profile:
            continue
        asset.primary_product = profile
        confirmation_required = confirmation_required or bool(profile.get("requires_user_confirmation"))

    if confirmation_required:
        task.status = TaskStatus.QUEUED
        task.workflow_stage = "primary_product_confirmation"
        task.workflow_message = "检测到多个相近商品，请确认本次视频需要推广的主商品。"
        task.workflow_progress = 0
        _append_workflow_event(task, task.workflow_stage, task.workflow_message, task.workflow_progress)
    task.updated_at = _utc_now()
    return repository.update(task)


def confirm_task_primary_product_selections(
    task_id: str,
    repository: InMemoryTaskRepository,
    selections: Dict[int, int],
) -> VideoTask:
    """保存用户选择的候选商品，确认完成后允许工作流继续启动。"""

    task = repository.get(task_id)
    for asset_index, asset in enumerate(task.uploaded_assets):
        profile = asset.primary_product
        if not profile.get("requires_user_confirmation"):
            continue
        if asset_index not in selections:
            raise TaskValidationError("请为每张存在歧义的素材选择一个主商品。")
        candidate_index = selections[asset_index]
        candidates = profile.get("candidates", [])
        if not isinstance(candidates, list) or candidate_index < 0 or candidate_index >= len(candidates):
            raise TaskValidationError("主商品候选不存在，请刷新页面后重新选择。")
        profile["selected_candidate_index"] = candidate_index
        profile["selection_method"] = "user_confirmed"
        profile["requires_user_confirmation"] = False

    task.workflow_stage = "created"
    task.workflow_message = "主商品已确认，准备启动工作流。"
    task.workflow_progress = 0
    _append_workflow_event(task, task.workflow_stage, task.workflow_message, task.workflow_progress)
    task.updated_at = _utc_now()
    return repository.update(task)


def start_task_workflow(task_id: str, repository: InMemoryTaskRepository) -> VideoTask:
    """
    启动任务对应的最小工作流占位逻辑。

    当前阶段不接真实 Agent 或模型能力，
    只把任务推进到一个明确的"工作流已开始"状态，
    这样页面和后续模块都能围绕这个状态继续演进。
    """

    print(f"[video_task_module] 准备启动工作流：task_id={task_id}", flush=True)
    logger.info("准备启动工作流：task_id=%s", task_id)

    task = repository.get(task_id)

    task.status = TaskStatus.PROCESSING
    task.workflow_stage = "planning"
    task.workflow_message = "工作流已自动启动，当前处于创作规划阶段。"
    task.workflow_progress = 5
    _append_workflow_event(task, "planning", task.workflow_message, task.workflow_progress)
    task.updated_at = _utc_now()

    print(
        "[video_task_module] 工作流状态更新完成："
        f"task_id={task.task_id}, status={task.status.value}, workflow_stage={task.workflow_stage}",
        flush=True,
    )
    logger.info(
        "工作流状态更新完成：task_id=%s, status=%s, workflow_stage=%s",
        task.task_id,
        task.status.value,
        task.workflow_stage,
    )
    return repository.update(task)


def finish_task_workflow(
    task_id: str,
    repository: InMemoryTaskRepository,
    workflow_result: Dict[str, Any],
) -> VideoTask:
    """
    保存工作流执行结果。

    这里不直接依赖 `agent/` 目录里的实现，保持任务模块只知道"结果字典"，
    不关心工作流内部用了哪些模型或步骤。
    """

    print(f"[video_task_module] 准备保存工作流结果：task_id={task_id}", flush=True)
    logger.info("准备保存工作流结果：task_id=%s", task_id)

    task = repository.get(task_id)

    task.workflow_result = workflow_result
    task.workflow_stage = str(workflow_result.get("workflow_stage", "draft_ready"))
    task.workflow_message = str(workflow_result.get("workflow_message", "工作流结果已生成。"))
    task.workflow_progress = int(workflow_result.get("workflow_progress", 100 if task.workflow_stage != "script_review" else 72))
    _append_workflow_event(task, task.workflow_stage, task.workflow_message, task.workflow_progress)
    task.updated_at = _utc_now()

    if workflow_result.get("workflow_status") == "completed":
        task.status = TaskStatus.COMPLETED
    elif workflow_result.get("workflow_status") == "needs_review":
        task.status = TaskStatus.NEEDS_REVIEW
    else:
        task.status = TaskStatus.PROCESSING

    _flow_print(
        "[video_task_module] 工作流结果保存完成："
        f"task_id={task.task_id}, status={task.status.value}, workflow_stage={task.workflow_stage}",
    )
    logger.info(
        "工作流结果保存完成：task_id=%s, status=%s, workflow_stage=%s",
        task.task_id,
        task.status.value,
        task.workflow_stage,
    )
    return repository.update(task)


def update_task_workflow_progress(
    task_id: str,
    repository: InMemoryTaskRepository,
    stage: str,
    message: str,
    progress: int,
) -> VideoTask:
    """保存后台工作流的阶段性进度，供详情页刷新展示。"""

    task = repository.get(task_id)
    task.status = TaskStatus.PROCESSING
    task.workflow_stage = stage
    task.workflow_message = message
    task.workflow_progress = max(0, min(99, int(progress)))
    _append_workflow_event(task, stage, message, task.workflow_progress)
    task.updated_at = _utc_now()
    return repository.update(task)


def update_task_workflow_partial(
    task_id: str,
    repository: InMemoryTaskRepository,
    partial_result: dict,
) -> VideoTask:
    """增量写入工作流中间产物，供前端实时展示剧本、分镜等阶段性结果。"""
    task = repository.get(task_id)
    existing = dict(task.workflow_result or {})
    existing.update(partial_result)
    task.workflow_result = existing
    task.updated_at = _utc_now()
    return repository.update(task)


def approve_task_script_review(
    task_id: str,
    repository: InMemoryTaskRepository,
    script_plan: Dict[str, Any],
    storyboard: List[Dict[str, Any]],
    script_review_variants: Dict[str, Any] | None = None,
    reviewer_note: str = "",
) -> VideoTask:
    """保存用户确认/编辑后的剧本分镜，并把任务推进到渲染阶段。"""

    task = repository.get(task_id)
    if task.workflow_stage != "script_review":
        raise TaskValidationError("当前任务不在剧本确认阶段，不能继续渲染。")
    if not script_plan:
        raise TaskValidationError("剧本内容不能为空。")
    if not storyboard:
        raise TaskValidationError("分镜内容不能为空。")

    result = dict(task.workflow_result or {})
    result["script_plan"] = script_plan
    result["storyboard"] = storyboard
    if script_review_variants:
        result["script_review_variants"] = script_review_variants
    result["script_review_user_note"] = reviewer_note.strip()
    result["workflow_stage"] = "render_video"
    result["workflow_status"] = "processing"
    result["workflow_message"] = "用户已确认剧本和分镜，正在继续生成视频。"
    task.workflow_result = result
    task.status = TaskStatus.PROCESSING
    task.workflow_stage = "render_video"
    task.workflow_message = "用户已确认剧本和分镜，正在继续生成视频。"
    task.workflow_progress = max(task.workflow_progress, 72)
    _append_workflow_event(task, task.workflow_stage, task.workflow_message, task.workflow_progress)
    task.updated_at = _utc_now()
    return repository.update(task)


def request_task_script_regeneration(
    task_id: str,
    repository: InMemoryTaskRepository,
    feedback: str,
) -> VideoTask:
    """记录用户对剧本的修改意见，并重新进入规划阶段。"""

    task = repository.get(task_id)
    if task.workflow_stage != "script_review":
        raise TaskValidationError("当前任务不在剧本确认阶段，不能重新生成剧本。")
    cleaned_feedback = feedback.strip()
    if not cleaned_feedback:
        raise TaskValidationError("请填写希望修改的方向。")

    task.chat_history.append(f"剧本重生成意见：{cleaned_feedback}")
    task.status = TaskStatus.PROCESSING
    task.workflow_stage = "planning"
    task.workflow_message = "已收到修改意见，正在重新生成剧本和分镜。"
    task.workflow_progress = 8
    task.workflow_result = {
        "workflow_status": "processing",
        "workflow_stage": "planning",
        "workflow_message": task.workflow_message,
        "regeneration_feedback": cleaned_feedback,
    }
    _append_workflow_event(task, task.workflow_stage, task.workflow_message, task.workflow_progress)
    task.updated_at = _utc_now()
    return repository.update(task)


def fail_task_workflow(
    task_id: str,
    repository: InMemoryTaskRepository,
    error_message: str,
) -> VideoTask:
    """后台工作流出现未捕获异常时，把任务标记为失败并保留错误原因。"""

    task = repository.get(task_id)
    task.status = TaskStatus.FAILED
    task.workflow_stage = "failed"
    task.workflow_message = error_message
    task.workflow_progress = max(task.workflow_progress, 1)
    task.workflow_result = {
        "workflow_status": "failed",
        "workflow_stage": "failed",
        "workflow_message": error_message,
    }
    _append_workflow_event(task, "failed", error_message, task.workflow_progress)
    task.updated_at = _utc_now()
    _flow_print(f"[video_task_module] 工作流失败：task_id={task.task_id}")
    return repository.update(task)


def _validate_create_command(command: CreateVideoTaskCommand) -> None:
    """校验创建任务时的最小输入规则。"""

    if not command.title.strip():
        raise TaskValidationError("任务标题不能为空。")

    if not command.selling_points:
        raise TaskValidationError("至少需要一个卖点。")

    cleaned_points = [point.strip() for point in command.selling_points if point.strip()]
    if not cleaned_points:
        raise TaskValidationError("卖点不能为空白内容。")

    if not command.target_platform.strip():
        raise TaskValidationError("目标平台不能为空。")

    if command.duration_seconds <= 0:
        raise TaskValidationError("视频时长必须大于 0。")

    if command.duration_seconds > 60:
        raise TaskValidationError("MVP 阶段的视频时长不能超过 60 秒。")

    if not command.style.strip():
        raise TaskValidationError("视频风格不能为空。")


def check_input_sufficiency(command: CreateVideoTaskCommand) -> dict[str, Any]:
    """检测用户输入信息是否足够支撑稳定的视频生成。"""

    signals: list[str] = []
    effective_text = "".join(
        [
            str(command.title),
            "".join(str(p) for p in command.selling_points),
            str(command.custom_style_prompt),
            str(command.product_type),
            str(command.target_audience),
            str(command.usage_scene),
            str(command.creative_direction),
            "".join(str(item) for item in command.forbidden_changes),
            "".join(str(item) for item in command.chat_history),
        ]
    )
    effective_chars = len("".join(effective_text.split()))

    if effective_chars < 30:
        confidence = "low"
        signals.append("text_total_too_brief")
    elif effective_chars < 60:
        confidence = "medium"
    else:
        confidence = "high"

    return {
        "confidence": confidence,
        "signals": signals,
        "should_warn": confidence == "low",
        "warning_message": (
            "当前信息较少，生成结果会更依赖模型自由发挥，可能出现商品外观、场景或卖点表达不稳定。"
            "你可以继续生成，也可以先补充 2-3 个问题来提升效果。"
        ) if confidence == "low" else "",
    }


def _generate_task_id() -> str:
    """生成任务对外使用的稳定标识符。"""

    return f"task_{uuid4().hex}"


def _append_workflow_event(
    task: VideoTask,
    stage: str,
    message: str,
    progress: int,
) -> None:
    """把阶段变化追加到任务事件流中，页面会用它展示后台执行过程。"""

    task.workflow_events.append(_workflow_event(stage, message, progress))


def _workflow_event(stage: str, message: str, progress: int) -> Dict[str, Any]:
    """生成一条轻量级工作流事件。"""

    return {
        "stage": stage,
        "message": message,
        "progress": progress,
        "created_at": _utc_now().isoformat(),
    }


def _utc_now() -> datetime:
    """统一时间戳生成逻辑，便于后续测试和替换。"""

    return datetime.now(timezone.utc)
