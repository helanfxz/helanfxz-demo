"""
Seedance 视频渲染模块。

这个模块只负责把创作计划提交给火山方舟视频生成接口。
它不决定分镜内容，也不决定每个分镜时长；这些都来自上游 LLM 生成的 storyboard。
"""

from __future__ import annotations

import base64
import builtins
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import chain
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

from agent.prompt_safety import (
    is_laptop_product,
    safe_text_to_video_scene_description,
    scene_text_mentions_recognizable_product,
)

VERBOSE_LOG = os.getenv("AIGC_VERBOSE_LOG") == "1"


def print(*args, **kwargs):  # type: ignore[override]
    """默认隐藏 Seedance 轮询细节输出。"""

    if VERBOSE_LOG:
        builtins.print(*args, **kwargs)


def _flow_print(message: str) -> None:
    """输出 Seedance 渲染关键结果。"""

    builtins.print(message, flush=True)


def render_seedance_video(
    task_id: str,
    creation_plan: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    """并行提交所有分镜到 Seedance，生成并合并最终视频。"""

    _flow_print(f"[seedance_video_renderer] 尝试调用 Seedance：task_id={task_id}")

    if os.getenv("AIGC_DISABLE_VIDEO_MODEL") == "1":
        return _render_result(False, "", "", [], "当前通过 AIGC_DISABLE_VIDEO_MODEL=1 禁用了视频模型调用。")

    api_key = os.getenv("ARK_API_KEY")
    model_endpoint = os.getenv("ARK_VIDEO_ENDPOINT_ID")
    if not api_key or not model_endpoint:
        return _render_result(False, "", "", [], "缺少 ARK_API_KEY 或 ARK_VIDEO_ENDPOINT_ID，无法调用 Seedance。")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    shots = creation_plan.get("shots", [])
    if not shots:
        return _render_result(False, "", "", [], "创作计划中没有分镜。")

    indexed_shots: list[tuple[int, dict[str, Any]]] = []
    shared_visual_style = creation_plan.get("visual_style_bible") or {}
    for i, shot in enumerate(shots):
        shot_index = int(shot.get("shot_index", i + 1))
        # 手写计划和历史任务也继承整片画风，不要求每个分镜重复保存相同字段。
        normalized_shot = dict(shot)
        if shared_visual_style and not normalized_shot.get("visual_style_bible"):
            normalized_shot["visual_style_bible"] = shared_visual_style
        indexed_shots.append((shot_index, normalized_shot))

    worker_limit = max(1, int(os.getenv("SEEDANCE_RENDER_WORKERS", str(os.cpu_count() or 4))))
    max_workers = min(len(indexed_shots), worker_limit)
    render_started_at = time.perf_counter()
    shot_results: list[dict[str, Any]] = []
    local_clip_paths: dict[int, Path] = {}
    model_indexed_shots: list[tuple[int, dict[str, Any]]] = []

    # 统一空背景只需要稳定承接，不交给生成模型改写场景。
    for shot_index, shot in indexed_shots:
        if not _should_render_scene_background_locally(shot):
            model_indexed_shots.append((shot_index, shot))
            continue
        local_clip_path = output_path / f"seedance_shot_{shot_index:02d}_local_scene.mp4"
        local_result = _render_local_scene_background_clip(shot, local_clip_path)
        shot_results.append({"shot_index": shot_index, **local_result})
        if local_result["success"]:
            local_clip_paths[shot_index] = local_clip_path
            _flow_print(f"[seedance_video_renderer] 本地场景承接镜完成：shot_index={shot_index}")
        else:
            # 本地生成失败时仍允许 Seedance 尝试，避免单个承接镜阻断整条视频。
            model_indexed_shots.append((shot_index, shot))
            _flow_print(
                "[seedance_video_renderer] 本地场景承接镜失败，改用 Seedance："
                f"shot_index={shot_index}, error={local_result['error']}"
            )

    # ---- Phase 1 + 2: 不同镜头组并行，同组续写镜头顺序执行 ----
    # 只有显式标记 continue_from_previous 的同组镜头才复用上一镜尾帧。
    render_batches = _split_seedance_render_batches(model_indexed_shots)
    submit_concurrency = max(1, int(os.getenv("SEEDANCE_SUBMIT_CONCURRENCY", "2")))
    _submit_semaphore = _create_bounded_semaphore(min(max_workers, submit_concurrency))
    _flow_print(
        "[seedance_video_renderer] 开始渲染镜头组："
        f"shot_count={len(model_indexed_shots)}, batch_count={len(render_batches)}, 最大提交并发={min(max_workers, submit_concurrency)}"
    )
    poll_results: dict[int, dict[str, Any]] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        batch_future_map = {
            pool.submit(
                _render_seedance_batch,
                api_key,
                model_endpoint,
                batch,
                _submit_semaphore,
            ): batch
            for batch in render_batches
        }
        for future in as_completed(batch_future_map):
            for task_result in future.result():
                elapsed = round(time.perf_counter() - render_started_at, 2)
                task_result["elapsed_seconds"] = elapsed
                shot_results.append(task_result)
                shot_index = int(task_result["shot_index"])
                if not task_result["success"]:
                    _flow_print(
                        "[seedance_video_renderer] Seedance 分镜任务失败："
                        f"shot_index={shot_index}, elapsed={elapsed:.2f}s, error={task_result['error']}"
                    )
                    continue
                poll_results[shot_index] = task_result
                _flow_print(
                    "[seedance_video_renderer] Seedance 分镜完成："
                    f"shot_index={shot_index}, seedance_task_id={task_result['seedance_task_id']}, elapsed={elapsed:.2f}s"
                )

    if not poll_results and not local_clip_paths:
        return _render_result(False, "", "", shot_results, "所有分镜提交 Seedance 均失败。")

    failed_count = sum(1 for r in shot_results if not r.get("success"))
    if not poll_results:
        return _render_result(
            False,
            "",
            "",
            sorted(shot_results, key=lambda item: int(item.get("shot_index", 0) or 0)),
            f"Seedance 所有 {failed_count} 个分镜均失败。",
        )

    required_indices = {
        int(shot.get("shot_index", index + 1))
        for index, shot in enumerate(shots)
        if shot.get("required_for_variant") is True
    }
    failed_required = [
        result
        for result in shot_results
        if int(result.get("shot_index", -1)) in required_indices and not result.get("success")
    ]
    if failed_required:
        failed_indexes = [int(item.get("shot_index", -1)) for item in failed_required]
        return _render_result(
            False,
            "",
            "",
            sorted(shot_results, key=lambda item: int(item.get("shot_index", 0) or 0)),
            f"required shot failed: {failed_indexes}",
        )

    # ---- Phase 3: 按分镜顺序下载视频并拼接 ----
    # 已有部分分镜成功，继续处理成功的分镜，失败分镜静默跳过。
    # 部分分镜下载失败不中断整体流程，跳过失败分镜继续处理成功的。
    clip_paths_by_index: dict[int, Path] = dict(local_clip_paths)
    shots_by_index = {int(shot.get("shot_index", index + 1)): shot for index, shot in enumerate(shots)}
    download_errors: list[str] = []
    for shot_index in sorted(poll_results):
        task_result = poll_results[shot_index]
        clip_path = output_path / f"seedance_shot_{shot_index:02d}.mp4"
        download_result = _download_video(task_result["video_url"], clip_path)
        for sr in shot_results:
            if sr.get("shot_index") == shot_index:
                sr["download"] = download_result
                break
        if not download_result["success"]:
            download_errors.append(f"shot {shot_index}: {download_result['error']}")
            _flow_print(
                "[seedance_video_renderer] 分镜下载失败，跳过："
                f"shot_index={shot_index}, error={download_result['error']}"
            )
            if shot_index in required_indices:
                return _render_result(
                    False,
                    "",
                    "",
                    sorted(shot_results, key=lambda item: int(item.get("shot_index", 0) or 0)),
                    f"required shot failed: [{shot_index}]",
                )
            continue

        # Seedance 1.5 固定生成 5 秒片段；这里按上游分镜目标时长裁剪后再拼接。
        adapted_clip_path = _adapt_clip_to_target_duration(
            source_path=clip_path,
            output_dir=output_path,
            shot_index=shot_index,
            shot=shots_by_index.get(shot_index, {}),
        )
        clip_paths_by_index[shot_index] = adapted_clip_path

    if not clip_paths_by_index:
        return _render_result(
            False, "", "", shot_results,
            f"所有分镜下载均失败：{'；'.join(download_errors)}",
        )

    ordered_shot_indices = sorted(clip_paths_by_index)
    clip_paths = [clip_paths_by_index[shot_index] for shot_index in ordered_shot_indices]
    raw_video_path = output_path / "seedance_raw.mp4"
    final_video_path = output_path / "seedance_final.mp4"
    transition_types = _transition_types_for_clip_order(shots_by_index, ordered_shot_indices)
    concat_result = _concat_videos(clip_paths, raw_video_path, transition_types=transition_types)
    if not concat_result["success"]:
        return _render_result(False, "", "", shot_results, concat_result["error"])

    subtitle_result = _overlay_storyboard_subtitles(
        source_video_path=raw_video_path,
        final_video_path=final_video_path,
        shots=creation_plan.get("shots", []),
    )
    if not subtitle_result["success"]:
        shutil.copyfile(raw_video_path, final_video_path)
        _flow_print(f"[seedance_video_renderer] 字幕叠加失败，保留无字幕视频：{subtitle_result['error']}")

    _flow_print(f"[seedance_video_renderer] Seedance 视频渲染完成：{final_video_path}")
    partial_error = (
        f"{len(download_errors)} 个分镜下载失败（{'；'.join(download_errors)}），"
        f"已使用 {len(clip_paths)} 个成功分镜合成视频。"
        if download_errors else None
    )
    return _render_result(
        success=True,
        video_path=str(final_video_path),
        video_url=_public_upload_url_for_video(final_video_path, task_id),
        shot_results=shot_results,
        error=partial_error,
        extra={"subtitle_overlay": subtitle_result},
    )


def _public_upload_url_for_video(video_path: Path, task_id: str) -> str:
    normalized = str(video_path).replace("\\", "/")
    marker = ".uploads/"
    if marker in normalized:
        return "/uploads/" + normalized.split(marker, 1)[1]
    return f"/uploads/{task_id}/{video_path.name}"


def _should_render_scene_background_locally(shot: dict[str, Any]) -> bool:
    """共享空背景镜使用本地稳定片段，避免视频模型自行增加商品或道具。"""

    asset = _resolve_seedance_asset(shot)
    return (
        str(shot.get("render_strategy", "")).strip() == "image_to_video"
        and bool(asset.get("file_path"))
        and bool(asset.get("is_scene_background"))
    )


def _render_local_scene_background_clip(shot: dict[str, Any], output_path: Path) -> dict[str, Any]:
    """把空棚拍底图转成轻量本地视频片段，完整保留首帧空间。"""

    asset = _resolve_seedance_asset(shot)
    source_path = Path(str(asset.get("file_path", "")))
    if not source_path.exists():
        return {"success": False, "error": f"共享场景底图不存在：{source_path}"}

    try:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image, ImageEnhance

        duration_seconds = float(shot.get("duration_seconds", 1) or 1)
        duration_seconds = max(1.0, duration_seconds)
        fps = 24
        frame_count = max(1, round(duration_seconds * fps))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        source = Image.open(source_path).convert("RGB").resize((720, 1280), Image.LANCZOS)
        reveal_path = Path(str(asset.get("reveal_asset_path", "")))
        reveal = (
            Image.open(reveal_path).convert("RGB").resize((720, 1280), Image.LANCZOS)
            if reveal_path.exists()
            else source
        )
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        try:
            for index in range(frame_count):
                # 同一背景中逐渐显出真实商品锚点，不移动构图，也不让模型新增物体。
                progress = index / max(1, frame_count - 1)
                reveal_progress = min(1.0, max(0.0, (progress - 0.10) / 0.80))
                frame = Image.blend(source, reveal, reveal_progress)
                brightness = 1.0 + 0.015 * progress
                frame = ImageEnhance.Brightness(frame).enhance(brightness)
                writer.append_data(np.asarray(frame))
        finally:
            writer.close()
    except Exception as exc:
        return {"success": False, "error": f"本地场景承接镜生成失败：{exc}"}

    return {
        "success": True,
        "status": "succeeded",
        "render_mode": "local_scene_background",
        "video_path": str(output_path),
        "error": None,
    }


def _render_local_identity_anchor_clip(shot: dict[str, Any], output_path: Path) -> dict[str, Any]:
    """把上传商品图转成轻量保真片段，避免模型再次重绘 Logo 和商品结构。"""

    asset = _resolve_seedance_asset(shot)
    source_path = Path(str(asset.get("file_path", "")))
    if not source_path.exists():
        return {"success": False, "error": f"商品保真锚点不存在：{source_path}"}

    try:
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image, ImageEnhance, ImageOps

        duration_seconds = max(1.0, float(shot.get("duration_seconds", 1) or 1))
        fps = 24
        frame_count = max(1, round(duration_seconds * fps))
        output_path.parent.mkdir(parents=True, exist_ok=True)

        source = ImageOps.contain(
            Image.open(source_path).convert("RGB"),
            (720, 1280),
            method=Image.LANCZOS,
        )
        canvas = Image.new("RGB", (720, 1280), (244, 244, 240))
        canvas.paste(source, ((720 - source.width) // 2, (1280 - source.height) // 2))
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)
        try:
            for index in range(frame_count):
                # 这里只调整极轻微亮度，不改变商品轮廓、Logo 像素结构或构图。
                progress = index / max(1, frame_count - 1)
                frame = ImageEnhance.Brightness(canvas).enhance(1.0 + 0.012 * progress)
                writer.append_data(np.asarray(frame))
        finally:
            writer.close()
    except Exception as exc:
        return {"success": False, "error": f"商品保真片段生成失败：{exc}"}

    return {
        "success": True,
        "status": "succeeded",
        "render_mode": "local_identity_anchor",
        "video_path": str(output_path),
        "clip_path": str(output_path),
        "error": None,
    }


def _create_bounded_semaphore(max_concurrent: int):
    """创建一个限制并发数的信号量。"""
    from threading import BoundedSemaphore
    return BoundedSemaphore(max_concurrent)


def _create_seedance_task_with_semaphore(
    semaphore,
    api_key: str,
    model_endpoint: str,
    shot: dict[str, Any],
    first_frame_url_override: str = "",
) -> dict[str, Any]:
    """带并发控制的 Seedance 任务创建。"""
    with semaphore:
        return _create_seedance_task(
            api_key,
            model_endpoint,
            shot,
            first_frame_url_override=first_frame_url_override,
        )


def _create_seedance_task(
    api_key: str,
    model_endpoint: str,
    shot: dict[str, Any],
    first_frame_url_override: str = "",
) -> dict[str, Any]:
    """创建单个分镜的视频生成任务。"""

    payload = _build_seedance_payload(model_endpoint, shot, first_frame_url_override)
    seedance_prompt = str(payload["content"][0]["text"])
    response = _send_json_request(
        method="POST",
        url="https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks",
        api_key=api_key,
        payload=payload,
        timeout=60,
    )
    if not response["success"]:
        response["seedance_prompt"] = seedance_prompt
        response["prompt_length"] = len(seedance_prompt)
        return response

    task_id = response["data"].get("id")
    if not task_id:
        return {
            "success": False,
            "error": f"Seedance 创建任务响应缺少 id：{response['data']}",
            "seedance_prompt": seedance_prompt,
            "prompt_length": len(seedance_prompt),
        }
    return {
        "success": True,
        "task_id": task_id,
        "error": None,
        "seedance_prompt": seedance_prompt,
        "prompt_length": len(seedance_prompt),
    }


def _build_seedance_payload(
    model_endpoint: str,
    shot: dict[str, Any],
    first_frame_url_override: str = "",
) -> dict[str, Any]:
    """构造单个分镜请求；续写镜头优先使用上一镜尾帧作为首帧。"""

    prompt_shot = dict(shot)
    if first_frame_url_override:
        prompt_shot["continuation_from_previous"] = True
    seedance_prompt = _build_seedance_prompt(prompt_shot)
    content = [
        {
            "type": "text",
            "text": seedance_prompt,
        }
    ]
    # render_input 是工作流交给渲染器的执行合同；asset 仅用于兼容旧任务。
    asset = _resolve_seedance_asset(shot)
    if first_frame_url_override:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": first_frame_url_override},
                "role": "first_frame",
            }
        )
        if (shot.get("anchor_last_frame") or shot.get("preserve_identity_tail")) and asset.get("file_path"):
            anchor_data_url = _image_file_to_data_url(asset["file_path"])
            if anchor_data_url:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": anchor_data_url},
                        "role": "last_frame",
                    }
                )
    elif shot.get("render_strategy") == "image_to_video" and asset.get("file_path"):
        data_url = _image_file_to_data_url(asset["file_path"])
        if data_url:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                    "role": "first_frame",
                }
            )
            if shot.get("preserve_identity_tail"):
                # 整机展示镜同时约束结尾，防止 Logo、键盘和机身在生成过程中逐步漂移。
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                        "role": "last_frame",
                    }
                )

    return {
        "model": model_endpoint,
        "content": content,
        "ratio": "9:16",
        "duration": _seedance_model_duration(shot),
        "resolution": "720p",
        "watermark": False,
        "generate_audio": False,
        # 后续镜头可以直接复用该尾帧续写，减少同一空间内的视觉跳变。
        "return_last_frame": True,
    }


def _resolve_seedance_asset(shot: dict[str, Any]) -> dict[str, Any]:
    """优先从 render_input 读取真实素材，兼容旧任务中的 asset 字段。"""

    render_input = shot.get("render_input") or {}
    if (
        isinstance(render_input, dict)
        and render_input.get("type") == "asset"
        and render_input.get("file_path")
    ):
        return {
            "asset_id": render_input.get("asset_id", ""),
            "asset_type": render_input.get("asset_type", "image"),
            "file_path": render_input["file_path"],
            "is_scene_background": bool(render_input.get("is_scene_background")),
            "reveal_asset_path": render_input.get("reveal_asset_path", ""),
        }
    asset = shot.get("asset") or {}
    return asset if isinstance(asset, dict) else {}


def _poll_seedance_task(api_key: str, seedance_task_id: str) -> dict[str, Any]:
    """轮询 Seedance 异步任务直到完成、失败或超时。"""

    max_wait_seconds = int(os.getenv("SEEDANCE_MAX_WAIT_SECONDS", "360"))
    interval_seconds = int(os.getenv("SEEDANCE_POLL_INTERVAL_SECONDS", "10"))
    deadline = time.time() + max_wait_seconds
    poll_started_at = time.perf_counter()
    last_status = "unknown"

    while time.time() < deadline:
        response = _send_json_request(
            method="GET",
            url=f"https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/{seedance_task_id}",
            api_key=api_key,
            payload=None,
            timeout=30,
        )
        if not response["success"]:
            return response

        data = response["data"]
        status = data.get("status")
        last_status = str(status)
        print(
            "[seedance_video_renderer] 查询 Seedance 任务："
            f"task_id={seedance_task_id}, status={status}",
            flush=True,
        )
        if status == "succeeded":
            video_url = (data.get("content") or {}).get("video_url")
            if not video_url:
                return {"success": False, "error": f"任务成功但缺少 video_url：{data}"}
            return {
                "success": True,
                "seedance_task_id": seedance_task_id,
                "status": status,
                "video_url": video_url,
                "last_frame_url": (data.get("content") or {}).get("last_frame_url", ""),
                "error": None,
            }
        if status in {"failed", "cancelled", "expired"}:
            return {
                "success": False,
                "seedance_task_id": seedance_task_id,
                "status": status,
                "error": f"Seedance 任务失败：{data}",
            }

        time.sleep(interval_seconds)

    return {
        "success": False,
        "seedance_task_id": seedance_task_id,
        "status": "timeout",
        "elapsed_seconds": round(time.perf_counter() - poll_started_at, 2),
        "last_status": last_status,
        "error": (
            f"Seedance 任务等待超过 {max_wait_seconds} 秒："
            f"seedance_task_id={seedance_task_id}, last_status={last_status}"
        ),
    }


def _should_continue_from_previous(
    shot: dict[str, Any],
    previous_shot: dict[str, Any] | None,
) -> bool:
    """仅让显式声明的同组镜头复用上一镜尾帧，避免跨场景串联和商品漂移累积。"""

    if not previous_shot:
        return False
    continuity_group = str(shot.get("continuity_group", "")).strip()
    previous_group = str(previous_shot.get("continuity_group", "")).strip()
    transition_type = str(shot.get("transition_type", "")).strip().lower()
    return bool(
        continuity_group
        and continuity_group == previous_group
        and transition_type == "continue_from_previous"
    )


def _transition_types_for_clip_order(
    shots_by_index: dict[int, dict[str, Any]],
    ordered_shot_indices: list[int],
) -> list[str]:
    """把分镜转场配置转换成片段边界配置；默认硬切，续写片段也直接衔接。"""

    transitions = []
    for shot_index in ordered_shot_indices[1:]:
        transition_type = str(
            shots_by_index.get(shot_index, {}).get("transition_type", "hard_cut")
        ).strip().lower()
        transitions.append("crossfade" if transition_type == "crossfade" else "hard_cut")
    return transitions


def _render_seedance_batch(
    api_key: str,
    model_endpoint: str,
    indexed_shots: list[tuple[int, dict[str, Any]]],
    submit_semaphore=None,
) -> list[dict[str, Any]]:
    """顺序渲染一个连续镜头组；显式续写时把上一镜尾帧作为下一镜首帧。"""

    results = []
    previous_shot: dict[str, Any] | None = None
    previous_last_frame_url = ""
    for shot_index, shot in indexed_shots:
        first_frame_url = (
            previous_last_frame_url
            if _should_continue_from_previous(shot, previous_shot)
            else ""
        )
        if submit_semaphore is None:
            create_result = _create_seedance_task(
                api_key,
                model_endpoint,
                shot,
                first_frame_url_override=first_frame_url,
            )
        else:
            create_result = _create_seedance_task_with_semaphore(
                submit_semaphore,
                api_key,
                model_endpoint,
                shot,
                first_frame_url_override=first_frame_url,
            )
        if not create_result["success"]:
            results.append({"shot_index": shot_index, **create_result})
            previous_shot = shot
            previous_last_frame_url = ""
            continue

        poll_result = _poll_seedance_task(api_key, create_result["task_id"])
        results.append({"shot_index": shot_index, **create_result, **poll_result})
        previous_shot = shot
        previous_last_frame_url = str(poll_result.get("last_frame_url", "")).strip()
    return results


def _split_seedance_render_batches(
    indexed_shots: list[tuple[int, dict[str, Any]]],
) -> list[list[tuple[int, dict[str, Any]]]]:
    """把显式续写镜头收进同一批次，其余镜头保持可并行执行。"""

    batches: list[list[tuple[int, dict[str, Any]]]] = []
    for indexed_shot in indexed_shots:
        _, shot = indexed_shot
        previous_shot = batches[-1][-1][1] if batches else None
        if batches and _should_continue_from_previous(shot, previous_shot):
            batches[-1].append(indexed_shot)
        else:
            batches.append([indexed_shot])
    return batches


def _build_seedance_prompt(shot: dict[str, Any]) -> str:
    """把单个分镜转换成适合 Seedance 的中文自然语言 prompt。"""

    forced_prompt = str(shot.get("video_prompt", "")).strip()
    if shot.get("force_video_prompt") and forced_prompt:
        return _limit_seedance_prompt(forced_prompt)

    render_strategy = str(shot.get("render_strategy", "")).strip()
    visual = _sanitize_seedance_generation_text(
        shot.get("visual_description", ""),
        render_strategy=render_strategy,
    )
    template_prompt = _sanitize_seedance_generation_text(
        shot.get("seedance_prompt", ""),
        render_strategy=render_strategy,
    )
    if template_prompt:
        visual = template_prompt
    if not visual:
        visual = " ".join(
            v
            for v in [
                _sanitize_seedance_generation_text(shot.get("scene_goal", ""), render_strategy=render_strategy),
                _sanitize_seedance_generation_text(shot.get("action", ""), render_strategy=render_strategy),
                _sanitize_seedance_generation_text(shot.get("final_state", ""), render_strategy=render_strategy),
                _sanitize_seedance_generation_text(shot.get("initial_state", ""), render_strategy=render_strategy),
            ]
            if v
        ).strip()
    purpose = _sanitize_seedance_generation_text(shot.get("purpose", ""), render_strategy=render_strategy)
    scene_goal = _sanitize_seedance_generation_text(shot.get("scene_goal", ""), render_strategy=render_strategy)
    identity_card = shot.get("product_identity_card") or {}
    motion_affordance = identity_card.get("motion_affordance") or {}
    product_presence = str(shot.get("product_presence", "optional")).strip().lower()
    continuation_from_previous = bool(shot.get("continuation_from_previous"))
    if render_strategy == "text_to_video" and (
        product_presence == "forbidden" or _text_to_video_requests_recognizable_product(shot)
    ):
        # 铺垫镜只负责建立场景。丢弃上游可能误写入的商品描述，避免生成错误近似商品。
        visual = _safe_text_to_video_scene_description(shot)
        product_presence = "forbidden"

    # 视频 prompt 约束优先使用 creation_plan 计算好的约束，其次从身份卡推导。
    vpc = shot.get("video_prompt_constraints") or {}
    if not isinstance(vpc, dict):
        vpc = {}
    extra_preserve = _clean_list(vpc.get("must_preserve", []))
    extra_avoid = _clean_list(vpc.get("must_avoid", []))

    # 动作约束：合并分镜动作和商品动作能力。
    action_text = _sanitize_seedance_generation_text(
        _safe_seedance_action(shot, render_strategy, product_presence),
        render_strategy=render_strategy,
    )
    forbidden_actions = _clean_list(motion_affordance.get("forbidden_actions", []))
    if forbidden_actions:
        action_text += "。禁止动作：" + "、".join(forbidden_actions)

    visible_marks = _clean_list(identity_card.get("visible_marks", []))
    text_avoid = (
        "新增的非商品自带文字、字符、字幕、水印、UI 元素或额外标签；不要改写商品自带 logo、标识或字样"
        if visible_marks
        else "画面内文字、字符、字幕、水印、UI 元素或额外标签"
    )
    avoid_parts = [
        text_avoid,
        "错误品牌标识、商品结构变形、翻转或不合理角度变化",
        "严禁出现烟雾、雾气、蒸汽、尘埃、光束或粒子飘浮",
    ]
    if product_presence == "required":
        avoid_parts.append("画面中只能保留一台商品主体，不新增第二台同类商品")
    avoid_parts.extend(extra_avoid)
    avoid_text = "；".join(avoid_parts)
    identity_text = _format_identity_constraints(identity_card)
    camera_text = _safe_seedance_camera_motion(shot, render_strategy)
    asset = _resolve_seedance_asset(shot)
    visual_style_text = _format_visual_style_bible(shot.get("visual_style_bible"))

    if continuation_from_previous:
        prompt = _structured_seedance_prompt({
            "生成方式": "首帧续写视频，第一帧来自上一镜尾帧，必须延续相同空间、构图、色温和光线方向",
            "镜头目标": purpose or scene_goal or "延续上一镜动作",
            "场景与构图": visual,
            "画面动作": action_text,
            "镜头动作": camera_text,
            "商品约束": (
                "沿用上一镜中的主体和道具；如果提供尾帧锚点，结尾必须自然回到尾帧中的真实商品外观。"
                "不凭空新增可识别商品、logo、品牌标识或可读文字"
            ),
            "必须避免": avoid_text,
            "整体风格": visual_style_text,
        })
    elif render_strategy == "image_to_video" and asset.get("file_path") and asset.get("is_scene_background"):
        prompt = _structured_seedance_prompt({
            "生成方式": "图生视频，使用统一场景底图作为首帧，只延续首帧中已有的背景",
            "镜头目标": purpose or "在统一背景上建立承接镜头",
            "场景连续性": "保持首帧的墙面、桌面、光线和构图，不新增商品主体，不新增家具或复杂道具",
            "画面动作": action_text,
            "镜头动作": camera_text,
            "必须避免": avoid_text + "；商品主体、logo、品牌标识或可读文字",
            "整体风格": visual_style_text,
        })
    elif render_strategy == "image_to_video" and asset.get("file_path"):
        prompt = _structured_seedance_prompt({
            "生成方式": "图生视频，使用上传素材作为首帧，后续画面必须延续首帧中的同一件商品",
            "镜头目标": purpose or scene_goal or "展示真实商品",
            "场景与构图": visual,
            "画面动作": action_text,
            "镜头动作": camera_text,
            "商品身份约束": identity_text + (
                "；补充保持：" + "、".join(extra_preserve)
                if extra_preserve else ""
            ),
            "必须避免": avoid_text,
            "整体风格": visual_style_text,
        })
    else:
        if product_presence == "required":
            product_constraint = identity_text + "；如果商品标识无法准确还原，不要突出或编造品牌标识"
        else:
            # 场景铺垫镜头不注入真实品牌细节，否则文生视频会尝试生成一个错误的近似商品。
            product_constraint = (
                "不展示可识别商品主体，不出现 logo、品牌标识或可读文字。"
                "如需出现商品，只允许无标识的模糊轮廓或远景剪影"
            )
        prompt = _structured_seedance_prompt({
            "生成方式": "文生视频，仅用于场景铺垫或不要求精确还原商品细节的镜头",
            "镜头目标": purpose or scene_goal or "建立商品使用场景",
            "场景与构图": visual,
            "画面动作": action_text,
            "镜头动作": camera_text,
            "商品约束": product_constraint,
            "必须避免": avoid_text,
            "整体风格": visual_style_text,
        })

    return _limit_seedance_prompt(prompt)


def _format_visual_style_bible(raw_style: Any) -> str:
    """把导演层的统一视觉语言压缩成每个分镜都必须继承的短提示词。"""

    if isinstance(raw_style, dict):
        summary = str(raw_style.get("style_summary") or raw_style.get("user_style") or "").strip()
        if summary:
            return "；".join(
                [
                    "真实写实的商业短视频",
                    summary,
                    "主体照明稳定",
                    "风格统一，背景不抢商品主体",
                    "稳定镜头，身份敏感镜头避免翻转和环绕",
                ]
            )
    defaults = {
        "realism": "真实写实的商业短视频",
        "lighting": "柔和自然光，主体照明稳定",
        "color_temperature": "中性偏暖色温",
        "background_complexity": "背景克制干净",
        "camera_language": "稳定镜头，运动幅度小",
    }
    if isinstance(raw_style, dict):
        for key in defaults:
            value = str(raw_style.get(key, "")).strip()
            if value:
                defaults[key] = value
    return "；".join(defaults.values())


def _safe_text_to_video_scene_description(shot: dict[str, Any]) -> str:
    """Compatibility wrapper for the shared prompt safety helper."""

    return safe_text_to_video_scene_description(shot)

def _scene_text_mentions_recognizable_product(text: str, shot: dict[str, Any]) -> bool:
    """Compatibility wrapper for the shared prompt safety helper."""

    return scene_text_mentions_recognizable_product(text, shot)


def _text_to_video_requests_recognizable_product(shot: dict[str, Any]) -> bool:
    """识别纯文生视频里会诱导模型凭空绘制品牌商品的描述。"""

    text = " ".join(
        str(shot.get(key, ""))
        for key in ("visual_description", "scene_goal", "action", "purpose")
    ).lower()
    brand_keywords = ("logo", "brand", "branded", "品牌", "商标", "标识")
    product_keywords = ("product", "商品", "laptop", "notebook", "笔记本", "电脑", "水杯", "杯子")
    return any(keyword in text for keyword in brand_keywords) and any(
        keyword in text for keyword in product_keywords
    )


def _safe_seedance_action(
    shot: dict[str, Any],
    render_strategy: str,
    product_presence: str,
) -> str:
    """收紧高风险商品动作，避免图生视频把真实商品改造成不合理结构。"""

    if product_presence == "forbidden":
        acting = str(shot.get("acting_direction", "")).strip()
        requested_action = str(shot.get("action", "")).strip()
        base_action = acting or requested_action
        if base_action and not _scene_text_mentions_recognizable_product(base_action, shot):
            return base_action + "。场景中不出现可识别商品主体或品牌标识。"
        return "保持场景稳定，只允许人物自然整理普通道具。"

    requested = str(shot.get("action") or "保持商品主体稳定").strip()
    identity_card = shot.get("product_identity_card") or {}
    product_type = str(identity_card.get("product_type", "")).strip()
    identity_strictness = str(shot.get("identity_strictness", "")).strip().lower()
    if render_strategy == "image_to_video" and identity_strictness == "high" and _is_laptop_product(product_type):
        return (
            "保持首帧中唯一一台笔记本电脑主体稳定，只允许轻微自然的光影变化；"
            "不要打开、合上、折叠、翻转或改变铰链角度，不新增第二台同类设备"
        )
    return requested


def _is_laptop_product(product_type: str) -> bool:
    """Compatibility wrapper for the shared prompt safety helper."""

    return is_laptop_product(product_type)


def _safe_seedance_camera_motion(shot: dict[str, Any], render_strategy: str) -> str:
    """按镜头风险收紧运镜，避免图生视频中的品牌标识和结构被补全变形。"""

    requested = str(shot.get("camera_motion", "")).strip()
    role = str(shot.get("narrative_role", "")).strip()
    identity_strictness = str(shot.get("identity_strictness", "")).strip().lower()
    review_text = " ".join(_clean_list(shot.get("review_focus", [])))
    content_text = " ".join(
        [
            str(shot.get("purpose", "")),
            str(shot.get("scene_goal", "")),
            str(shot.get("visual_description", "")),
            review_text,
        ]
    ).lower()
    is_identity_detail = role == "detail_proof" or identity_strictness == "high"
    is_identity_detail = is_identity_detail or any(
        keyword in content_text for keyword in ("logo", "商标", "品牌标识", "材质", "结构")
    )

    # detail_proof 或 high strictness：无论渲染模式，强制定镜
    if is_identity_detail:
        return "定镜，保持构图稳定，不推近、不拉远、不旋转、不环绕"

    if render_strategy == "image_to_video":
        return "仅允许轻微水平平移，幅度不超过画面 10%，不推近、不拉远、不旋转、不环绕"
    if any(keyword in requested for keyword in ("旋转", "环绕", "翻转", "推近", "推入")):
        return "轻微水平平移，幅度不超过画面 15%"
    return requested or "定镜或轻微水平平移，保持构图稳定"


def _format_identity_constraints(identity_card: dict[str, Any]) -> str:
    """把商品身份卡压缩成适合放进视频 prompt 的约束文本。"""

    if not identity_card:
        return "保持用户上传素材中的商品主体、颜色、结构和真实外观，不要生成其他品类商品。"

    appearance = str(identity_card.get("appearance_summary", "")).strip()
    must_preserve = [str(i) for i in identity_card.get("must_preserve", []) if str(i).strip()]
    forbidden = [str(i) for i in identity_card.get("forbidden_changes", []) if str(i).strip()]
    visible_marks = [str(i) for i in identity_card.get("visible_marks", []) if str(i).strip()]

    parts = [appearance] if appearance and appearance != "unknown" else []
    if visible_marks:
        parts.append("保持首帧已有商品标识稳定，不新增、改写或重新绘制文字")
    if must_preserve:
        parts.append("必须保持：" + "、".join(must_preserve))
    if forbidden:
        parts.append("禁止变化：" + "、".join(forbidden))

    result = "；".join(parts)
    if not result or result == "unknown":
        return "保持用户上传素材中的商品主体、颜色、结构和真实外观，不要生成其他品类商品。"
    return result


def _clean_list(items: list) -> list[str]:
    """清洗字符串列表，去空白去重。"""
    seen = set()
    result = []
    for item in items:
        s = str(item).strip()
        if s and s not in seen:
            seen.add(s)
            result.append(s)
    return result


def _sanitize_seedance_generation_text(value: Any, *, render_strategy: str) -> str:
    """清理不应交给视频模型绘制的文字和 UI 指令。"""

    text = str(value or "").strip()
    if not text:
        return ""

    # CTA 字幕、按钮和图标由本地后处理负责。继续留在视频 prompt 中会诱导模型生成乱码。
    text = re.sub(
        r"(?:随后|然后|并且)?画面(?:定格，?)?(?:出现|叠加)[^。；]*(?:文字|字幕|图标|按钮|UI)[^。；]*[。；]?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?:随后|然后|并且)?叠加[^。；]*(?:文字|字幕|图标|按钮|UI)[^。；]*[。；]?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # 文生视频无法准确还原指定品牌文字。图生视频也只允许保持首帧已有标识，不能要求模型重新绘制。
    clauses = re.split(r"([，。；])", text)
    cleaned_parts: list[str] = []
    mark_guard_added = False
    for clause in clauses:
        if not clause or clause in {"，", "。", "；"}:
            if cleaned_parts and clause:
                cleaned_parts.append(clause)
            continue
        lower_clause = clause.lower()
        has_mark_request = any(keyword in lower_clause for keyword in ("logo", "品牌标识", "商标", "字样"))
        if not has_mark_request:
            cleaned_parts.append(clause)
            continue
        if render_strategy == "image_to_video" and not mark_guard_added:
            cleaned_parts.append("保持首帧已有商品标识稳定，不新增、改写或重新绘制文字")
            mark_guard_added = True

    return "".join(cleaned_parts).strip(" ，。；")


def _clean_seedance_text(value: Any) -> str:
    """清洗发给 Seedance 的文本，避免把品牌/字幕/JSON 痕迹带进 content.text。"""

    text = str(value or "")
    replacements = {
        "\n": " ",
        "\r": " ",
        "{": " ",
        "}": " ",
        "[": " ",
        "]": " ",
        "\"": "",
        "'": "",
        "logo文字": "logo text",
        "logoæ–‡å­—": "logo text",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return " ".join(text.split())


def _structured_seedance_prompt(fields: dict[str, Any]) -> str:
    """把视频 prompt 固定成分区自然语言段落，并按字段压缩。"""

    # 之前统一 900 字符截断会把后面的商品身份约束或必须避免项截掉。
    # 这里先按字段截断，再用总长限制兜底，保证 Scene / Product / Action / Camera 都能保留。
    section_limits = {
        "生成方式": 120,
        "镜头目标": 160,
        "场景与构图": 360,
        "场景连续性": 240,
        "画面动作": 260,
        "镜头动作": 140,
        "商品约束": 360,
        "商品身份约束": 460,
        "必须避免": 320,
        "整体风格": 180,
    }

    lines = []
    for key, value in fields.items():
        cleaned = _clean_seedance_text(value)
        if not cleaned:
            continue
        limit = section_limits.get(str(key), 240)
        cleaned = _truncate_seedance_section(cleaned, limit)
        if cleaned:
            lines.append(f"{key}：{cleaned}")
    return "\n".join(lines)


def _truncate_seedance_section(text: str, max_chars: int) -> str:
    """按字段压缩 prompt，优先在中文/英文标点处截断。"""

    text = " ".join(str(text).split())
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    for sep in ("；", "。", ",", "，", " "):
        if sep in clipped:
            head = clipped.rsplit(sep, 1)[0]
            if len(head) >= max(40, int(max_chars * 0.55)):
                return head
    return clipped


def _limit_seedance_prompt(prompt: str, max_chars: int = 1600) -> str:
    """限制 Seedance prompt 总长度，同时避免粗暴截断关键约束。"""

    lines = [" ".join(line.split()) for line in str(prompt).splitlines()]
    prompt = "\n".join(line for line in lines if line)
    if len(prompt) <= max_chars:
        return prompt

    # 保留靠前字段和末尾的“必须避免/整体风格”，避免身份约束被截没。
    kept = []
    total = 0
    for line in lines:
        if not line:
            continue
        projected = total + len(line) + 1
        if projected <= max_chars:
            kept.append(line)
            total = projected
        elif line.startswith(("必须避免", "整体风格")):
            compressed = _truncate_seedance_section(line, max(120, max_chars - total - 1))
            if compressed:
                kept.append(compressed)
            break
    return "\n".join(kept) or prompt[:max_chars]


def _seedance_model_duration(shot: dict[str, Any]) -> int:
    """把 LLM 分镜时长适配成 Seedance 1.5 当前可接受的片段时长。"""

    # 课题说明里提到 Seedance 1.5 片段按 5 秒生成。
    # 上游 storyboard 仍保留 LLM 决策时长；这里仅是模型接口适配层。
    return 5


def _target_render_duration(shot: dict[str, Any]) -> float:
    """读取分镜目标时长，用于裁剪 Seedance 片段和计算字幕时间轴。"""

    render_segment = shot.get("render_segment") or {}
    value = render_segment.get("target_duration_seconds", shot.get("duration_seconds", 5))
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return float(_seedance_model_duration(shot))


def _send_json_request(
    method: str,
    url: str,
    api_key: str,
    payload: dict[str, Any] | None,
    timeout: int,
) -> dict[str, Any]:
    """发送火山方舟 JSON 请求。"""

    max_attempts = int(os.getenv("SEEDANCE_REQUEST_RETRIES", "3"))
    last_error = ""

    for attempt in range(1, max_attempts + 1):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else b"{}"
        req = request.Request(
            url,
            data=body if method != "GET" else None,
            method=method,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(req, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            return {"success": True, "data": data, "error": None}
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="ignore")
            # 4xx 多半是参数错误，重试没有意义；429 是频率限制，等待后重试。
            if 400 <= exc.code < 500 and exc.code != 429:
                return {"success": False, "error": f"Seedance HTTP 错误：status={exc.code}, body={error_body}"}
            last_error = f"Seedance HTTP 错误：status={exc.code}, body={error_body}"
        except URLError as exc:
            last_error = f"Seedance 网络错误：{exc.reason}"
        except TimeoutError:
            last_error = "Seedance 请求超时。"
        except socket.gaierror as exc:
            last_error = f"Seedance DNS 解析失败：{exc}"
        except json.JSONDecodeError as exc:
            return {"success": False, "error": f"Seedance 响应不是合法 JSON：{exc}"}

        if attempt < max_attempts:
            _flow_print(f"[seedance_video_renderer] 网络请求失败，准备重试：attempt={attempt}, error={last_error}")
            time.sleep(8 * attempt if "429" in last_error or "RateLimit" in last_error else 2 * attempt)

    return {"success": False, "error": last_error}


def _download_video(video_url: str, output_path: Path) -> dict[str, Any]:
    """下载 Seedance 返回的视频文件。"""

    max_attempts = int(os.getenv("SEEDANCE_DOWNLOAD_RETRIES", "3"))
    last_error = ""
    for attempt in range(1, max_attempts + 1):
        try:
            with request.urlopen(video_url, timeout=120) as response:
                output_path.write_bytes(response.read())
            return {"success": True, "path": str(output_path), "error": None}
        except Exception as exc:
            last_error = f"下载视频失败：{exc}"
            if attempt < max_attempts:
                _flow_print(
                    "[seedance_video_renderer] 分镜下载失败，准备重试："
                    f"attempt={attempt}, error={last_error}"
                )
                time.sleep(2 * attempt)
    return {"success": False, "error": last_error}


def _adapt_clip_to_target_duration(
    source_path: Path,
    output_dir: Path,
    shot_index: int,
    shot: dict[str, Any],
) -> Path:
    """按分镜目标时长裁剪模型生成片段，失败时保留原片段。"""

    target_duration = _target_render_duration(shot)
    model_duration = float(_seedance_model_duration(shot))
    if target_duration >= model_duration:
        return source_path

    if (shot.get("anchor_last_frame") or shot.get("preserve_identity_tail")) and target_duration >= model_duration - 0.25:
        # 首尾帧约束片段必须保留结尾真实锚点；直接截取前 N 秒会把最重要的回落过程裁掉。
        retimed_path = output_dir / f"seedance_shot_{shot_index:02d}_retimed.mp4"
        retime_result = _retime_video(
            source_path,
            retimed_path,
            duration_seconds=target_duration,
            source_duration_seconds=model_duration,
        )
        if retime_result["success"]:
            _flow_print(
                "[seedance_video_renderer] 首尾帧片段整体变速完成："
                f"shot_index={shot_index}, target_duration={target_duration:.2f}s"
            )
            return retimed_path
        _flow_print(
            "[seedance_video_renderer] 首尾帧片段整体变速失败，回退普通裁剪："
            f"shot_index={shot_index}, error={retime_result['error']}"
        )

    trimmed_path = output_dir / f"seedance_shot_{shot_index:02d}_trimmed.mp4"
    trim_result = _trim_video(source_path, trimmed_path, target_duration)
    if trim_result["success"]:
        _flow_print(
            "[seedance_video_renderer] 分镜片段裁剪完成："
            f"shot_index={shot_index}, target_duration={target_duration:.2f}s"
        )
        return trimmed_path

    _flow_print(
        "[seedance_video_renderer] 分镜片段裁剪失败，保留原片段："
        f"shot_index={shot_index}, error={trim_result['error']}"
    )
    return source_path


def _trim_video(source_path: Path, output_path: Path, duration_seconds: float) -> dict[str, Any]:
    """调用 ffmpeg 裁剪单个分镜片段。"""

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        return {"success": False, "error": f"缺少 imageio-ffmpeg，无法裁剪视频：{exc}"}

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(source_path),
        "-t",
        f"{duration_seconds:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {"success": False, "error": completed.stderr}
    return {"success": True, "error": None}


def _retime_video(
    source_path: Path,
    output_path: Path,
    duration_seconds: float,
    source_duration_seconds: float,
) -> dict[str, Any]:
    """压缩完整视频片段到目标时长，保留首帧和尾帧锚点。"""

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        return {"success": False, "error": f"缺少 imageio-ffmpeg，无法调整视频速度：{exc}"}

    speed_ratio = duration_seconds / source_duration_seconds
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i",
        str(source_path),
        "-vf",
        f"setpts={speed_ratio:.6f}*PTS",
        "-r",
        "24",
        "-t",
        f"{duration_seconds:.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {"success": False, "error": completed.stderr}
    return {"success": True, "error": None}


def _concat_videos(
    clip_paths: list[Path],
    final_video_path: Path,
    transition_types: list[str] | None = None,
) -> dict[str, Any]:
    """按边界配置合并分镜；默认硬切，只有显式要求时才使用交叉溶解。"""

    if not clip_paths:
        return {"success": False, "error": "没有可合并的视频片段。"}
    if len(clip_paths) == 1:
        shutil.copyfile(clip_paths[0], final_video_path)
        return {"success": True, "error": None}

    normalized_transitions = transition_types or ["hard_cut"] * (len(clip_paths) - 1)
    if not any(item == "crossfade" for item in normalized_transitions):
        return _concat_videos_without_transition(clip_paths, final_video_path)

    transition_result = _concat_videos_with_frame_crossfade(
        clip_paths,
        final_video_path,
        transition_types=normalized_transitions,
    )
    if transition_result["success"]:
        return transition_result
    _flow_print(
        "[seedance_video_renderer] 逐帧交叉溶解失败，回退为普通拼接："
        f"{transition_result['error']}"
    )
    return _concat_videos_without_transition(clip_paths, final_video_path)


def _concat_videos_with_frame_crossfade(
    clip_paths: list[Path],
    final_video_path: Path,
    transition_seconds: float = 0.45,
    transition_types: list[str] | None = None,
) -> dict[str, Any]:
    """仅在指定边界逐帧叠化；其他边界保留短视频常用的直接切镜。"""

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        return {"success": False, "error": f"缺少 imageio，无法执行逐帧叠化：{exc}"}

    writer = None
    pending_tail: list[Any] = []
    fps = 24.0
    transition_frames = max(1, int(round(fps * transition_seconds)))

    try:
        for clip_index, clip_path in enumerate(clip_paths):
            reader = imageio.get_reader(clip_path)
            try:
                metadata = reader.get_meta_data()
                if clip_index == 0:
                    fps = float(metadata.get("fps") or 24)
                    transition_frames = max(1, int(round(fps * transition_seconds)))
                    writer = imageio.get_writer(final_video_path, fps=fps, codec="libx264", quality=8)

                frames = iter(reader)
                if clip_index > 0:
                    transition_type = (
                        transition_types[clip_index - 1]
                        if transition_types and clip_index - 1 < len(transition_types)
                        else "hard_cut"
                    )
                    if transition_type == "crossfade":
                        next_head = []
                        for _ in range(transition_frames):
                            try:
                                next_head.append(next(frames))
                            except StopIteration:
                                break
                        for blended in _blend_transition_frames(pending_tail, next_head):
                            writer.append_data(blended)
                        # 下一镜头开头仍然保留，确保总时长和字幕时间轴不变。
                        pending_tail = []
                        frames = chain(next_head, frames)
                    else:
                        for frame in pending_tail:
                            writer.append_data(frame)
                        pending_tail = []

                for frame in frames:
                    pending_tail.append(frame)
                    if len(pending_tail) > transition_frames:
                        writer.append_data(pending_tail.pop(0))
            finally:
                reader.close()

        if writer is None:
            return {"success": False, "error": "没有可用视频帧。"}
        for frame in pending_tail:
            writer.append_data(frame)
    except Exception as exc:
        return {"success": False, "error": str(exc)}
    finally:
        if writer is not None:
            writer.close()

    return {"success": True, "error": None}


def _blend_transition_frames(previous_tail: list[Any], next_head: list[Any]) -> list[Any]:
    """把前一镜尾部与后一镜开头逐帧混合，避免边界出现硬切或黑屏。"""

    import numpy as np

    frame_count = min(len(previous_tail), len(next_head))
    if frame_count == 0:
        return list(previous_tail)
    previous_tail = previous_tail[-frame_count:]
    blended_frames = []
    for index, (previous, following) in enumerate(zip(previous_tail, next_head)):
        alpha = (index + 1) / (frame_count + 1)
        blended = previous.astype(np.float32) * (1 - alpha) + following.astype(np.float32) * alpha
        blended_frames.append(np.clip(blended, 0, 255).astype(np.uint8))
    return blended_frames


def _concat_videos_without_transition(clip_paths: list[Path], final_video_path: Path) -> dict[str, Any]:
    """转场处理不可用时执行普通拼接，确保主链路仍能返回视频。"""

    try:
        import imageio_ffmpeg
    except ImportError as exc:
        return {"success": False, "error": f"缺少 imageio-ffmpeg，无法合并视频：{exc}"}

    concat_file = final_video_path.with_suffix(".txt")
    concat_file.write_text(
        "\n".join(f"file '{path.resolve().as_posix()}'" for path in clip_paths),
        encoding="utf-8",
    )
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c",
        "copy",
        str(final_video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        return {"success": False, "error": f"视频合并失败：{completed.stderr}"}
    return {"success": True, "error": None}


def _overlay_storyboard_subtitles(
    source_video_path: Path,
    final_video_path: Path,
    shots: list[dict[str, Any]],
) -> dict[str, Any]:
    """把分镜字幕按时间轴烧录到 Seedance 视频上。"""

    subtitle_lines = _build_subtitle_timeline(shots)
    if not subtitle_lines:
        shutil.copyfile(source_video_path, final_video_path)
        return {"success": True, "mode": "skipped", "error": None}

    try:
        return _overlay_subtitles_with_pillow(source_video_path, final_video_path, subtitle_lines)
    except Exception as exc:
        return {"success": False, "mode": "pillow_subtitles", "error": f"字幕叠加失败：{exc}"}


def _overlay_subtitles_with_pillow(
    source_video_path: Path,
    final_video_path: Path,
    subtitle_lines: list[dict[str, Any]],
) -> dict[str, Any]:
    """用 Pillow 逐帧绘制字幕，直接指定中文字体文件，避免 fontconfig 找不到中文字体。"""

    try:
        from PIL import Image, ImageDraw, ImageFont
        import imageio.v2 as imageio
    except ImportError as exc:
        raise RuntimeError(f"缺少字幕渲染依赖：{exc}") from exc

    reader = imageio.get_reader(source_video_path)
    metadata = reader.get_meta_data()
    fps = float(metadata.get("fps") or 24)
    font_path = _subtitle_font_path()
    font_factory = lambda size: ImageFont.truetype(font_path, size)
    frames = []

    for frame_index, frame in enumerate(reader):
        timestamp = frame_index / fps
        subtitle = _subtitle_at_time(subtitle_lines, timestamp)
        image = Image.fromarray(frame).convert("RGB")
        if subtitle:
            _draw_subtitle_box(ImageDraw.Draw(image), image.size, subtitle, font_factory)
        frames.append(image)

    reader.close()
    imageio.mimsave(final_video_path, frames, fps=fps, quality=8)

    return {
        "success": True,
        "mode": "pillow_subtitles",
        "font_path": font_path,
        "error": None,
    }


def _build_subtitle_timeline(shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """根据分镜顺序生成字幕时间轴；时间长度按 Seedance 实际片段长度计算。"""

    timeline: list[dict[str, Any]] = []
    start_seconds = 0.0
    for shot in shots:
        duration_seconds = _target_render_duration(shot)
        subtitle = str(shot.get("subtitle", "")).strip()
        if subtitle:
            timeline.append(
                {
                    "start": start_seconds,
                    "end": start_seconds + duration_seconds,
                    "text": subtitle,
                }
            )
        start_seconds += duration_seconds
    return timeline


def _build_ass_content(subtitle_lines: list[dict[str, Any]], font_name: str) -> str:
    """生成 ASS 字幕内容，后续交给 ffmpeg 烧录到视频画面。"""

    events = []
    for item in subtitle_lines:
        text = _escape_ass_text(_wrap_subtitle_text(str(item["text"]), max_chars=14))
        events.append(
            "Dialogue: 0,"
            f"{_format_ass_time(float(item['start']))},"
            f"{_format_ass_time(float(item['end']))},"
            f"Default,,0,0,0,,{text}"
        )

    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 720",
            "PlayResY: 1280",
            "",
            "[V4+ Styles]",
            (
                "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
                "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
                "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
            ),
            (
                f"Style: Default,{font_name},48,&H00FFFFFF,&H00FFFFFF,&H7F000000,&H7F000000,"
                "0,0,0,0,100,100,0,0,1,3,1,2,56,56,120,1"
            ),
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
            *events,
            "",
        ]
    )


def _subtitle_font_name() -> str:
    """选择 ffmpeg ASS 字幕优先使用的中文字体名称。"""

    # 真实字体是否存在由运行环境决定；没有中文字体时，视频仍会生成，但中文可能显示异常。
    # 推荐安装：sudo apt install -y fonts-noto-cjk fonts-wqy-microhei
    return os.getenv("AIGC_SUBTITLE_FONT", "Noto Sans CJK SC")


def _subtitle_font_path() -> str:
    """选择可直接被 Pillow 加载的中文字体文件。"""

    configured_path = os.getenv("AIGC_SUBTITLE_FONT_PATH", "").strip()
    candidates = [
        configured_path,
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
    ]
    for font_path in candidates:
        if font_path and Path(font_path).exists():
            return font_path
    raise RuntimeError("未找到可用中文字体，请安装 fonts-noto-cjk 或设置 AIGC_SUBTITLE_FONT_PATH。")


def _subtitle_at_time(subtitle_lines: list[dict[str, Any]], timestamp: float) -> str:
    """根据当前时间找到应该显示的字幕。"""

    for item in subtitle_lines:
        if float(item["start"]) <= timestamp < float(item["end"]):
            return str(item["text"])
    return ""


def _draw_subtitle_box(draw, image_size: tuple[int, int], subtitle: str, font_factory) -> None:
    """在视频底部绘制半透明字幕底和可读字幕。"""

    width, height = image_size
    box_left = 48
    box_right = width - 48
    max_text_width = max(120, box_right - box_left - 48)
    max_lines = 5
    font = font_factory(44)
    lines = _wrap_plain_subtitle_text_by_width(draw, subtitle, font, max_text_width, max_lines=max_lines)
    while _subtitle_lines_too_wide(draw, lines, font, max_text_width) and getattr(font, "size", 44) > 26:
        font = font_factory(max(26, int(getattr(font, "size", 44)) - 2))
        lines = _wrap_plain_subtitle_text_by_width(draw, subtitle, font, max_text_width, max_lines=max_lines)

    line_height = max(40, int(getattr(font, "size", 44) * 1.32))
    box_height = 34 + line_height * len(lines)
    box_top = height - box_height - 80
    box_bottom = height - 80

    draw.rounded_rectangle(
        (box_left, box_top, box_right, box_bottom),
        radius=18,
        fill=(0, 0, 0),
        outline=(255, 255, 255),
        width=2,
    )
    text_y = box_top + 17
    for line in lines:
        text_width = _text_width(draw, line, font)
        draw.text(((width - text_width) / 2, text_y), line, fill=(255, 255, 255), font=font)
        text_y += line_height


def _wrap_plain_subtitle_text(text: str, max_chars: int) -> list[str]:
    """按字符数换行，返回 Pillow 绘制使用的纯文本行。"""

    return [text[index : index + max_chars] for index in range(0, len(text), max_chars)] or [""]


def _wrap_plain_subtitle_text_by_width(
    draw,
    text: str,
    font,
    max_width: int,
    *,
    max_lines: int,
) -> list[str]:
    """按实际像素宽度换行，尽量完整显示中文短句。"""

    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    if not cleaned:
        return [""]

    lines: list[str] = []
    current = ""
    for char in cleaned:
        candidate = current + char
        if current and _text_width(draw, candidate, font) > max_width:
            lines.append(current.rstrip())
            current = char
            if len(lines) >= max_lines:
                break
        else:
            current = candidate
    if current and len(lines) < max_lines:
        lines.append(current.rstrip())

    if len(lines) > max_lines:
        lines = lines[:max_lines]
    return lines or [cleaned]


def _subtitle_lines_too_wide(draw, lines: list[str], font, max_width: int) -> bool:
    """判断字幕行是否仍超过可用宽度。"""

    return any(_text_width(draw, line, font) > max_width for line in lines)


def _text_width(draw, text: str, font) -> int:
    """兼容不同 Pillow 版本的文本宽度计算。"""

    if hasattr(draw, "textbbox"):
        left, _, right, _ = draw.textbbox((0, 0), text, font=font)
        return right - left
    return int(draw.textlength(text, font=font))


def _wrap_subtitle_text(text: str, max_chars: int) -> str:
    """粗略按中文字符数换行，避免字幕过长挤出画面。"""

    return "\\N".join(text[index : index + max_chars] for index in range(0, len(text), max_chars))


def _format_ass_time(seconds: float) -> str:
    """把秒数格式化成 ASS 需要的 h:mm:ss.cc。"""

    centiseconds = int(round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def _escape_ass_text(text: str) -> str:
    """转义 ASS 字幕里的特殊字符。"""

    return text.replace("{", r"\{").replace("}", r"\}")


def _escape_ffmpeg_filter_path(path: Path) -> str:
    """转义 ffmpeg filter 参数中的路径分隔符。"""

    return str(path.resolve()).replace("\\", "/").replace(":", r"\:")


def _image_file_to_data_url(image_path: str) -> str:
    """把本地图片转成接口可接收的 data URL。"""

    path = Path(image_path)
    if not path.exists():
        return ""
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def repair_and_rerender_shot(
    shot: dict[str, Any],
    shot_index: int,
    repair_strategy: str,
    task_id: str,
    output_dir: str,
) -> dict[str, Any]:
    """根据修复策略重新渲染单个分镜。

    支持三类单镜修复：
    - rerender_with_stronger_identity_anchor：强化商品身份后重渲染；
    - simplify_action：收敛复杂动作后重渲染；
    - rewrite_shot_goal：以分镜目标为准重写当前镜头后重渲染。
    """
    if repair_strategy == "fallback_to_local_identity_anchor":
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        clip_path = output_path / f"seedance_shot_{shot_index:02d}_local_identity.mp4"
        _flow_print(f"[seedance_video_renderer] 使用上传素材生成保真片段：shot_{shot_index}")
        return _render_local_identity_anchor_clip(shot, clip_path)

    supported_strategies = {
        "rerender_with_stronger_identity_anchor",
        "simplify_action",
        "rewrite_shot_goal",
    }
    if repair_strategy not in supported_strategies:
        return {
            "success": False,
            "error": (
                f"不支持的修复策略：{repair_strategy}"
                "（支持 rerender_with_stronger_identity_anchor、simplify_action、rewrite_shot_goal）"
            ),
        }

    api_key = os.getenv("ARK_API_KEY")
    model_endpoint = os.getenv("ARK_VIDEO_ENDPOINT_ID")
    if not api_key or not model_endpoint:
        return {"success": False, "error": "缺少 API 密钥或端点，无法重试。"}

    shot = _prepare_repair_shot(dict(shot), repair_strategy)

    _flow_print(f"[seedance_video_renderer] 修复重渲染 shot_{shot_index}...")
    create_result = _create_seedance_task(api_key, model_endpoint, shot)
    if not create_result.get("success"):
        return {"success": False, "error": create_result.get("error", "创建修复任务失败")}

    seedance_task_id = create_result.get("task_id", "")
    poll_result = _poll_seedance_task(api_key, seedance_task_id)
    if not poll_result.get("success"):
        return {"success": False, "error": poll_result.get("error", "轮询修复任务失败")}

    video_url = poll_result.get("video_url", "")
    if not video_url:
        return {"success": False, "error": "修复任务未返回视频地址"}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    clip_path = output_path / f"seedance_shot_{shot_index:02d}_repaired.mp4"
    download_ok = _download_video(video_url, clip_path)
    if not download_ok:
        return {"success": False, "error": "修复视频下载失败"}

    # 裁剪到目标时长
    adapted = _adapt_clip_to_target_duration(
        source_path=clip_path,
        output_dir=output_path,
        shot_index=shot_index,
        shot=shot,
    )
    if adapted.exists():
        clip_path = adapted

    _flow_print(f"[seedance_video_renderer] 修复 shot_{shot_index} 完成：{clip_path}")
    return {"success": True, "clip_path": str(clip_path), "video_url": video_url}


def _prepare_repair_shot(shot: dict[str, Any], repair_strategy: str) -> dict[str, Any]:
    """把审核策略转换为当前分镜可直接执行的重渲染 Prompt。"""

    if repair_strategy == "simplify_action":
        # 动作失败通常来自过复杂的人物、液体或结构变化；先把镜头收敛成稳定展示。
        shot["action"] = "保持主体稳定，只允许轻微自然光影变化和缓慢镜头推近，不倒水、不旋转、不打开结构"
        shot["camera_motion"] = "slow stable push in"
    elif repair_strategy == "rewrite_shot_goal":
        scene_goal = str(shot.get("scene_goal") or shot.get("purpose") or "").strip()
        if scene_goal:
            shot["visual_description"] = scene_goal
        shot["action"] = "只执行分镜目标中的一个简单、清晰、可拍动作，保持场景元素干净稳定"
        shot["camera_motion"] = "stable establishing shot"

    base_prompt = _build_seedance_prompt(shot)
    identity_card = shot.get("product_identity_card") or {}
    must_preserve = list(identity_card.get("must_preserve", []))
    visible_marks = list(identity_card.get("visible_marks", []))
    if visible_marks:
        constraints = [
            "Repair this single shot only.",
            "No newly generated non-product text, no subtitles, no watermark, no UI.",
            "Keep existing product marks only; do not redraw, rewrite, or invent letters.",
            "Keep scene elements minimal and consistent with the shot goal.",
            "Avoid liquid spill, impossible product motion, duplicated objects, and structure deformation.",
        ]
        constraints.append("Preserve only brand marks already visible in the input frame; do not redraw or rewrite text.")
    else:
        constraints = [
            "Repair this single shot only.",
            "No generated text, no glyphs, no subtitles, no watermark, no UI.",
            "Keep scene elements minimal and consistent with the shot goal.",
            "Avoid liquid spill, impossible product motion, duplicated objects, and structure deformation.",
        ]
    if must_preserve:
        constraints.append("Must preserve: " + ", ".join(must_preserve) + ".")
    if repair_strategy == "rewrite_shot_goal":
        constraints.append("Use the rewritten shot goal as the source of truth; ignore previous wrong scene details.")
    if repair_strategy == "simplify_action":
        constraints.append("Simplify the action to one stable movement; prioritize correctness over creativity.")

    shot["video_prompt"] = " ".join(constraints) + "\n" + base_prompt
    shot["force_video_prompt"] = True
    return shot


def _render_result(
    success: bool,
    video_path: str,
    video_url: str,
    shot_results: list[dict[str, Any]],
    error: str | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """统一 Seedance 渲染结果结构。"""

    result = {
        "render_mode": "seedance",
        "success": success,
        "video_path": video_path,
        "video_url": video_url,
        "shot_results": shot_results,
        "error": error,
    }
    if extra:
        result.update(extra)
    return result
