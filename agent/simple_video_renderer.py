"""
本地视频预览渲染模块。

这个模块只负责把已经生成的分镜草稿合成为一个可播放的 MP4。
它不调用视频生成模型，也不负责剧本、分镜或素材理解。
"""

from __future__ import annotations

import builtins
import os
from pathlib import Path
from typing import Any

VERBOSE_LOG = os.getenv("AIGC_VERBOSE_LOG") == "1"


def print(*args, **kwargs):  # type: ignore[override]
    """默认隐藏本地渲染细节输出。"""

    if VERBOSE_LOG:
        builtins.print(*args, **kwargs)


def _flow_print(message: str) -> None:
    """输出本地预览关键结果。"""

    builtins.print(message, flush=True)


def render_preview_video(
    task_id: str,
    storyboard: list[dict[str, Any]],
    asset_matching: list[dict[str, Any]],
    output_dir: str,
) -> dict[str, Any]:
    """根据分镜和图片素材生成本地 MP4 预览。"""

    print(f"[simple_video_renderer] 开始生成本地预览视频：task_id={task_id}", flush=True)

    try:
        from PIL import Image, ImageDraw, ImageFont
        import imageio.v2 as imageio
    except ImportError as exc:
        return _render_result(
            success=False,
            video_path="",
            video_url="",
            error=f"缺少视频渲染依赖：{exc}",
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    video_path = output_path / "preview.mp4"

    image_assets = _collect_image_assets(asset_matching)
    if not storyboard:
        return _render_result(False, "", "", "没有分镜，无法生成视频。")
    try:
        font = _load_font(ImageFont)
        frames = []
        fps = 1

        for index, shot in enumerate(storyboard):
            asset_path = image_assets[index % len(image_assets)] if image_assets else ""
            duration_seconds = max(1, int(shot.get("duration_seconds", 1)))
            subtitle = str(shot.get("subtitle", "")).strip()
            storyboard_text = _build_storyboard_preview_text(shot)

            # 本地 fallback 保留 LLM 决定的分镜时长。
            # 有素材就展示素材，没有素材就生成文字占位画面。
            frame = _build_vertical_frame(
                image_module=Image,
                draw_module=ImageDraw,
                font=font,
                image_path=asset_path,
                storyboard_text=storyboard_text,
                subtitle=subtitle,
            )
            frames.extend([frame] * duration_seconds * fps)

        # imageio 会通过 imageio-ffmpeg 写出 MP4，不依赖系统 PATH 里的 ffmpeg。
        imageio.mimsave(video_path, frames, fps=fps, quality=8)
    except Exception as exc:
        return _render_result(False, "", "", f"本地预览视频生成失败：{exc}")

    _flow_print(f"[simple_video_renderer] 本地预览视频生成完成：{video_path}")
    return _render_result(
        success=True,
        video_path=str(video_path),
        video_url=f"/uploads/{task_id}/preview.mp4",
        error=None,
    )


def _collect_image_assets(asset_matching: list[dict[str, Any]]) -> list[str]:
    """从素材匹配结果中提取可用于本地预览的图片路径。"""

    image_paths: list[str] = []
    for item in asset_matching:
        asset = item.get("matched_asset") or {}
        if asset.get("asset_type") == "image" and asset.get("file_path"):
            image_paths.append(str(asset["file_path"]))
    return image_paths


def _build_vertical_frame(
    image_module,
    draw_module,
    font,
    image_path: str,
    storyboard_text: str,
    subtitle: str,
) -> Any:
    """生成单个 9:16 竖版视频帧。"""

    canvas_width = 720
    canvas_height = 1280
    canvas = image_module.new("RGB", (canvas_width, canvas_height), (248, 246, 240))

    draw = draw_module.Draw(canvas)
    draw.text((56, 48), "本地预览分镜", fill=(17, 24, 39), font=font)
    if image_path and Path(image_path).exists():
        source = image_module.open(image_path).convert("RGB")
        source.thumbnail((canvas_width - 96, 500))
        image_x = (canvas_width - source.width) // 2
        image_y = 120
        canvas.paste(source, (image_x, image_y))
    else:
        draw.rounded_rectangle(
            (72, 120, canvas_width - 72, 600),
            radius=24,
            fill=(230, 238, 244),
            outline=(150, 166, 180),
            width=2,
        )
        text_y = 220
        for line in _wrap_text(storyboard_text or "分镜描述缺失", max_chars=16)[:6]:
            draw.text((104, text_y), line, fill=(31, 41, 55), font=font)
            text_y += 58

    # 分镜说明是模型失败后的关键兜底信息；有商品图时也必须显示。
    draw.rounded_rectangle(
        (48, 650, canvas_width - 48, 900),
        radius=18,
        fill=(255, 255, 255),
        outline=(210, 205, 196),
        width=2,
    )
    text_y = 680
    for line in _wrap_text(storyboard_text or "分镜描述缺失", max_chars=18)[:4]:
        draw.text((72, text_y), line, fill=(31, 41, 55), font=font)
        text_y += 52

    subtitle_lines = _wrap_text(subtitle, max_chars=18)
    text_y = 970

    # 字幕区域用浅色底，保证不同商品图上文字都能读。
    draw.rounded_rectangle(
        (48, text_y - 24, canvas_width - 48, canvas_height - 96),
        radius=18,
        fill=(255, 255, 255),
        outline=(210, 205, 196),
        width=2,
    )
    for line in subtitle_lines[:4]:
        draw.text((72, text_y), line, fill=(32, 33, 36), font=font)
        text_y += 58

    return canvas


def _build_storyboard_preview_text(shot: dict[str, Any]) -> str:
    """把结构化分镜压缩成回退视频里可读的说明文字。"""

    parts = [
        f"镜头 {shot.get('shot_index', '')}".strip(),
        str(shot.get("scene_goal") or shot.get("purpose") or "").strip(),
        str(shot.get("initial_state", "")).strip(),
        str(shot.get("action", "")).strip(),
        str(shot.get("final_state", "")).strip(),
        str(shot.get("visual_description", "")).strip(),
    ]
    return " / ".join(part for part in parts if part)


def _wrap_text(text: str, max_chars: int) -> list[str]:
    """按中文短句粗略换行，避免字幕超出画面。"""

    if not text:
        return ["暂无字幕"]
    return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]


def _load_font(image_font_module):
    """优先加载常见中文字体，失败时使用默认字体。"""

    candidates = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/mnt/c/Windows/Fonts/msyh.ttc",
        "/mnt/c/Windows/Fonts/simhei.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            return image_font_module.truetype(font_path, 36)
    return image_font_module.load_default()


def _render_result(
    success: bool,
    video_path: str,
    video_url: str,
    error: str | None,
) -> dict[str, Any]:
    """统一渲染结果结构。"""

    return {
        "render_mode": "local_preview",
        "success": success,
        "video_path": video_path,
        "video_url": video_url,
        "error": error,
    }
