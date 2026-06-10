"""
素材预处理管线。

对用户上传的图片做统一修复和标准化，确保进入下游流水线的素材质量一致。
不依赖 LLM，全部是确定性操作。
"""

from __future__ import annotations

from functools import lru_cache
from io import BytesIO
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import numpy as np

TARGET_WIDTH = 720
TARGET_HEIGHT = 1280
TARGET_RATIO = TARGET_WIDTH / TARGET_HEIGHT  # 9:16

SHARPNESS_THRESHOLD = 100.0  # 拉普拉斯方差低于此值视为模糊
BRIGHTNESS_MIN = 40          # 平均亮度低于此值视为过暗
BRIGHTNESS_MAX = 220         # 平均亮度高于此值视为过曝


def preprocess_asset(
    image_path: str,
    output_dir: str,
    selected_candidate_index: int | None = None,
) -> dict[str, Any]:
    """
    对单张图片做完整的预处理流水线，输出标准化后的图片路径和诊断信息。
    """

    original = Image.open(image_path).convert("RGB")
    diagnostics: dict[str, Any] = {
        "original_path": image_path,
        "original_size": original.size,
    }

    # 1. 清晰度修复（轻度锐化 + 降噪）
    laplacian_var = _estimate_sharpness(original)
    diagnostics["sharpness_score"] = round(laplacian_var, 1)
    if laplacian_var < SHARPNESS_THRESHOLD:
        original = _enhance_sharpness(original)
        diagnostics["sharpness_fixed"] = True
    else:
        diagnostics["sharpness_fixed"] = False

    # 2. 曝光修正
    mean_brightness = _mean_brightness(original)
    diagnostics["brightness"] = round(mean_brightness, 1)
    if mean_brightness < BRIGHTNESS_MIN:
        original = _adjust_exposure(original, factor=1.4)
        diagnostics["exposure_fixed"] = "underexposed"
    elif mean_brightness > BRIGHTNESS_MAX:
        original = _adjust_exposure(original, factor=0.7)
        diagnostics["exposure_fixed"] = "overexposed"
    else:
        diagnostics["exposure_fixed"] = False

    # 3. 白平衡校正
    original = _auto_white_balance(original)

    # 抠图使用修复后的原始比例图片，避免先裁剪导致商品边缘被截断。
    foreground_source = original.copy()

    # 4. 画幅适配：裁剪到 9:16，保留一份标准化原图作为分析和降级素材。
    original = _fit_to_aspect_ratio(original, TARGET_RATIO)
    diagnostics["after_crop_size"] = original.size

    # 5. 缩放到目标分辨率
    original = original.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)

    # 6. 保存标准化原图。多模态分析优先看这张图，保留真实拍摄上下文。
    output_path = Path(output_dir) / f"preprocessed_{Path(image_path).stem}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original.save(output_path, format="JPEG", quality=92)
    diagnostics["output_path"] = str(output_path)

    # 7. 生成统一背景锚点图。视频渲染优先使用它，避免不同上传照片背景直接跳变。
    keyframe_source_path = output_path
    try:
        foreground = _remove_background(foreground_source)
        # rembg 只负责提取前景。画面中存在多个商品时，还需要先选出一个主商品，
        # 避免把辅助商品一起写入后续视频生成使用的外观锚点。
        foreground, primary_product = _select_primary_foreground(
            foreground,
            output_dir=output_dir,
            stem=Path(image_path).stem,
            selected_candidate_index=selected_candidate_index,
        )
        diagnostics["primary_product"] = primary_product
        if not _foreground_is_usable(foreground):
            raise RuntimeError("抠图结果主体过于透明，已回退标准化原图。")
        anchor = _compose_studio_anchor(foreground)
        anchor_path = Path(output_dir) / f"anchor_{Path(image_path).stem}.jpg"
        anchor.save(anchor_path, format="JPEG", quality=94)
        diagnostics["background_removed"] = True
        diagnostics["anchor_output_path"] = str(anchor_path)
        keyframe_source_path = anchor_path
    except Exception as exc:
        diagnostics["background_removed"] = False
        diagnostics["background_removal_error"] = str(exc)

    # 8. 构造弱关键帧：不依赖图片生成 API，仅用裁剪、重排和留白制造不同镜头锚点。
    try:
        diagnostics["keyframe_variants"] = _create_keyframe_variants(
            image_path=str(keyframe_source_path),
            output_dir=output_dir,
            stem=Path(image_path).stem,
        )
    except Exception as exc:
        diagnostics["keyframe_variant_error"] = str(exc)
        diagnostics["keyframe_variants"] = {}

    return diagnostics


def preprocess_all_assets(assets: list[dict[str, Any]], output_dir: str) -> list[dict[str, Any]]:
    """批量预处理素材，返回每个素材的诊断信息。"""

    results: list[dict[str, Any]] = []
    for asset in assets:
        file_path = asset.get("file_path", "")
        if not file_path or not Path(file_path).exists():
            results.append({"original_path": file_path, "error": "文件不存在"})
            continue
        if asset.get("asset_type") != "image":
            results.append({"original_path": file_path, "error": "非图片类型，跳过"})
            continue
        try:
            selected_candidate_index = asset.get("primary_product", {}).get("selected_candidate_index")
            diag = preprocess_asset(file_path, output_dir, selected_candidate_index=selected_candidate_index)
            results.append(diag)
        except Exception as exc:
            results.append({"original_path": file_path, "error": str(exc)})
    return results



def _create_keyframe_variants(image_path: str, output_dir: str, stem: str) -> dict[str, str]:
    """
    基于已有商品图构造不同分镜可用的弱关键帧。

    这些图片不是生成式改图，而是确定性的重构：
    - hero：商品完整展示，适合主视觉 / product_reveal；
    - detail：中心区域放大，适合材质、logo 或结构细节证明；
    - cta：商品上移并保留下方字幕安全区，适合结尾引导。
    """

    output_root = Path(output_dir) / "keyframes"
    output_root.mkdir(parents=True, exist_ok=True)
    source = Image.open(image_path).convert("RGB").resize((TARGET_WIDTH, TARGET_HEIGHT), Image.LANCZOS)

    hero_path = output_root / f"hero_{stem}.jpg"
    source.save(hero_path, format="JPEG", quality=94)

    detail = _center_crop_zoom(source, zoom=1.45)
    detail_path = output_root / f"detail_{stem}.jpg"
    detail.save(detail_path, format="JPEG", quality=94)

    cta = _compose_cta_frame(source)
    cta_path = output_root / f"cta_{stem}.jpg"
    cta.save(cta_path, format="JPEG", quality=94)

    return {
        "hero": str(hero_path),
        "detail": str(detail_path),
        "cta": str(cta_path),
    }


def _center_crop_zoom(image: Image.Image, zoom: float = 1.35) -> Image.Image:
    """从中心裁剪并放大，作为不依赖生成模型的细节镜头首帧。"""

    width, height = image.size
    crop_w = int(width / max(1.01, zoom))
    crop_h = int(height / max(1.01, zoom))
    left = max(0, (width - crop_w) // 2)
    top = max(0, (height - crop_h) // 2)
    crop = image.crop((left, top, left + crop_w, top + crop_h))
    return crop.resize((width, height), Image.LANCZOS)


def _compose_cta_frame(image: Image.Image) -> Image.Image:
    """构造下方留白的结尾定格帧，避免 CTA 字幕遮挡商品。"""

    width, height = image.size
    canvas = _build_studio_background((width, height))
    product = image.copy()
    product.thumbnail((int(width * 0.78), int(height * 0.58)), Image.LANCZOS)
    x = (width - product.width) // 2
    y = int(height * 0.08)
    canvas.paste(product, (x, y))

    draw = ImageDraw.Draw(canvas)
    safe_top = int(height * 0.72)
    draw.rectangle((0, safe_top, width, height), fill=(228, 220, 207))
    draw.line((int(width * 0.12), safe_top + int(height * 0.03), int(width * 0.88), safe_top + int(height * 0.03)), fill=(196, 184, 166), width=max(1, width // 240))
    return canvas

def create_studio_background(output_dir: str) -> str:
    """生成可复用的空棚拍场景底图，让相邻镜头共享同一个视觉空间。"""

    output_path = Path(output_dir) / "studio_background.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _build_studio_background().save(output_path, format="JPEG", quality=94)
    return str(output_path)


# ---- 内部工具函数 ----


def _estimate_sharpness(image: Image.Image) -> float:
    """拉普拉斯方差估计图像清晰度。"""
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float64)
    laplacian = np.abs(np.gradient(np.gradient(arr, axis=0), axis=0) +
                       np.gradient(np.gradient(arr, axis=1), axis=1))
    return float(laplacian.var())


def _enhance_sharpness(image: Image.Image) -> Image.Image:
    """轻度锐化 + 降噪。"""
    # UnsharpMask: radius=2, percent=150 模拟
    blurred = image.filter(ImageFilter.GaussianBlur(radius=1.5))
    sharpened = Image.blend(image, blurred, alpha=-0.3)  # type: ignore[arg-type]
    return sharpened


def _mean_brightness(image: Image.Image) -> float:
    """计算图像平均亮度。"""
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.float64)
    return float(arr.mean())


def _adjust_exposure(image: Image.Image, factor: float) -> Image.Image:
    """调整曝光。factor > 1 提亮，< 1 压暗。"""
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def _auto_white_balance(image: Image.Image) -> Image.Image:
    """
    简单白平衡校正：假设画面中最亮的 5% 像素是白色参考，
    对各通道做线性缩放使参考点变为纯白。
    """
    arr = np.array(image, dtype=np.float64)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # 取亮度最高的 5% 像素作为白点参考
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    threshold = np.percentile(gray, 95)
    mask = gray >= threshold

    if mask.sum() < 100:
        return image  # 参考像素太少，跳过

    r_ref = r[mask].mean()
    g_ref = g[mask].mean()
    b_ref = b[mask].mean()
    ref_avg = (r_ref + g_ref + b_ref) / 3.0

    if ref_avg < 1:
        return image

    scale_r = ref_avg / r_ref if r_ref > 0 else 1.0
    scale_g = ref_avg / g_ref if g_ref > 0 else 1.0
    scale_b = ref_avg / b_ref if b_ref > 0 else 1.0

    arr[:, :, 0] = np.clip(r * scale_r, 0, 255)
    arr[:, :, 1] = np.clip(g * scale_g, 0, 255)
    arr[:, :, 2] = np.clip(b * scale_b, 0, 255)

    return Image.fromarray(arr.astype(np.uint8))


def _fit_to_aspect_ratio(image: Image.Image, target_ratio: float) -> Image.Image:
    """
    将图片适配到目标宽高比。优先居中裁剪，如果图片不够大则留黑边。
    """
    w, h = image.size
    current_ratio = w / h

    if abs(current_ratio - target_ratio) < 0.02:
        return image

    if current_ratio > target_ratio:
        # 图片偏宽：裁剪左右
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        return image.crop((left, 0, left + new_w, h))

    # 图片偏高：裁剪上下
    new_h = int(w / target_ratio)
    top = (h - new_h) // 2
    return image.crop((0, top, w, top + new_h))


@lru_cache(maxsize=1)
def _rembg_session():
    """复用抠图模型会话，避免每张图片重复加载模型。"""

    from rembg import new_session

    return new_session("u2net", providers=["CPUExecutionProvider"])


def _remove_background(image: Image.Image) -> Image.Image:
    """使用 rembg 去除图片背景，输出带透明通道的商品前景。"""

    if os.getenv("AIGC_DISABLE_BACKGROUND_REMOVAL") == "1":
        raise RuntimeError("当前通过 AIGC_DISABLE_BACKGROUND_REMOVAL=1 禁用了抠图。")

    from rembg import remove

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    removed = remove(buffer.getvalue(), session=_rembg_session())
    foreground = Image.open(BytesIO(removed)).convert("RGBA")
    if not foreground.getchannel("A").getbbox():
        raise RuntimeError("抠图结果没有有效商品前景。")
    return foreground


def _select_primary_foreground(
    foreground: Image.Image,
    output_dir: str,
    stem: str,
    selected_candidate_index: int | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """从透明前景中选出最可能的主商品，并保存后续可复用的 mask。"""

    rgba = foreground.convert("RGBA")
    alpha = np.array(rgba.getchannel("A"))
    candidates = _extract_foreground_candidates(alpha)
    if not candidates:
        raise RuntimeError("抠图结果中没有面积足够的商品候选区域。")

    user_confirmed = selected_candidate_index is not None
    if selected_candidate_index is None:
        selected_candidate_index = 0
    if selected_candidate_index < 0 or selected_candidate_index >= len(candidates):
        raise ValueError("用户选择的主商品候选不存在。")
    selected = candidates[selected_candidate_index]
    selected_alpha = np.where(selected["component_mask"], alpha, 0).astype(np.uint8)
    selected_rgba = np.array(rgba)
    selected_rgba[:, :, 3] = selected_alpha
    selected_foreground = Image.fromarray(selected_rgba)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    mask_path = output_root / f"primary_product_mask_{stem}.png"
    foreground_path = output_root / f"primary_product_{stem}.png"
    Image.fromarray(selected_alpha).save(mask_path, format="PNG")
    selected_foreground.save(foreground_path, format="PNG")

    relative_margin = 1.0
    if len(candidates) > 1:
        relative_margin = (selected["score"] - candidates[1]["score"]) / max(selected["score"], 0.001)
    requires_confirmation = not user_confirmed and len(candidates) > 1 and relative_margin < 0.20
    confidence = 0.98 if len(candidates) == 1 else min(0.95, 0.55 + max(0.0, relative_margin))

    profile = {
        "bbox": selected["bbox"],
        "mask_path": str(mask_path),
        "foreground_path": str(foreground_path),
        "confidence": round(confidence, 2),
        "selection_method": "user_confirmed" if user_confirmed else "automatic_area_and_center",
        "selected_candidate_index": selected_candidate_index,
        "candidate_count": len(candidates),
        "requires_user_confirmation": requires_confirmation,
        "source_size": list(rgba.size),
        "candidates": [
            {
                "bbox": candidate["bbox"],
                "score": round(candidate["score"], 3),
                "area_ratio": round(candidate["area_ratio"], 4),
            }
            for candidate in candidates
        ],
    }
    return selected_foreground, profile


def _extract_foreground_candidates(alpha: np.ndarray) -> list[dict[str, Any]]:
    """按透明通道的连通区域拆分商品候选，并按面积和中心位置排序。"""

    try:
        from scipy import ndimage
    except ImportError:
        return _extract_foreground_candidates_without_scipy(alpha)

    visible = alpha >= 96
    labels, count = ndimage.label(visible)
    if count == 0:
        return []

    height, width = visible.shape
    total_pixels = height * width
    min_pixels = max(100, int(total_pixels * 0.002))
    image_center_x = width / 2
    image_center_y = height / 2
    max_center_distance = max(1.0, (image_center_x ** 2 + image_center_y ** 2) ** 0.5)

    candidates: list[dict[str, Any]] = []
    for component_id, component_slice in enumerate(ndimage.find_objects(labels), start=1):
        if component_slice is None:
            continue
        component_mask = labels == component_id
        pixel_count = int(component_mask.sum())
        if pixel_count < min_pixels:
            continue

        y_slice, x_slice = component_slice
        bbox = [x_slice.start, y_slice.start, x_slice.stop, y_slice.stop]
        center_x = (x_slice.start + x_slice.stop) / 2
        center_y = (y_slice.start + y_slice.stop) / 2
        center_distance = ((center_x - image_center_x) ** 2 + (center_y - image_center_y) ** 2) ** 0.5
        center_score = max(0.0, 1.0 - center_distance / max_center_distance)
        area_ratio = pixel_count / total_pixels
        area_score = min(1.0, area_ratio / 0.40)

        candidates.append({
            "bbox": bbox,
            "component_mask": component_mask,
            "area_ratio": area_ratio,
            "score": 0.75 * area_score + 0.25 * center_score,
        })

    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _extract_foreground_candidates_without_scipy(alpha: np.ndarray) -> list[dict[str, Any]]:
    """在没有 scipy 时用简单 flood fill 提取透明通道连通区域。"""

    visible = alpha >= 96
    height, width = visible.shape
    if not visible.any():
        return []

    total_pixels = height * width
    min_pixels = max(100, int(total_pixels * 0.002))
    image_center_x = width / 2
    image_center_y = height / 2
    max_center_distance = max(1.0, (image_center_x ** 2 + image_center_y ** 2) ** 0.5)
    visited = np.zeros_like(visible, dtype=bool)
    candidates: list[dict[str, Any]] = []

    ys, xs = np.nonzero(visible)
    for seed_y, seed_x in zip(ys.tolist(), xs.tolist()):
        if visited[seed_y, seed_x]:
            continue
        stack = [(seed_y, seed_x)]
        visited[seed_y, seed_x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and visible[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))

        pixel_count = len(pixels)
        if pixel_count < min_pixels:
            continue

        comp_ys = [p[0] for p in pixels]
        comp_xs = [p[1] for p in pixels]
        x0, x1 = min(comp_xs), max(comp_xs) + 1
        y0, y1 = min(comp_ys), max(comp_ys) + 1
        component_mask = np.zeros_like(visible, dtype=bool)
        component_mask[comp_ys, comp_xs] = True
        center_x = (x0 + x1) / 2
        center_y = (y0 + y1) / 2
        center_distance = ((center_x - image_center_x) ** 2 + (center_y - image_center_y) ** 2) ** 0.5
        center_score = max(0.0, 1.0 - center_distance / max_center_distance)
        area_ratio = pixel_count / total_pixels
        area_score = min(1.0, area_ratio / 0.40)
        candidates.append({
            "bbox": [x0, y0, x1, y1],
            "component_mask": component_mask,
            "area_ratio": area_ratio,
            "score": 0.75 * area_score + 0.25 * center_score,
        })

    return sorted(candidates, key=lambda item: item["score"], reverse=True)


def _foreground_is_usable(foreground: Image.Image) -> bool:
    """判断抠图结果是否保留了足够完整的不透明商品主体。"""

    alpha = foreground.convert("RGBA").getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return False
    pixels = list(alpha.crop(bbox).getdata())
    if not pixels:
        return False
    solid_ratio = sum(value >= 128 for value in pixels) / len(pixels)
    return solid_ratio >= 0.15


def _compose_studio_anchor(
    foreground: Image.Image,
    target_size: tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT),
) -> Image.Image:
    """把透明商品前景放到统一棚拍背景，减少跨镜头背景跳变。"""

    width, height = target_size
    canvas = _build_studio_background(target_size)
    table_top = int(height * 0.72)

    rgba = foreground.convert("RGBA")
    alpha_bbox = rgba.getchannel("A").getbbox()
    if not alpha_bbox:
        raise ValueError("透明前景中没有可见商品。")
    product = rgba.crop(alpha_bbox)
    product.thumbnail((int(width * 0.84), int(height * 0.64)), Image.LANCZOS)

    # 商品底部落在桌面区域，避免悬浮感。
    x = (width - product.width) // 2
    y = min(table_top - int(product.height * 0.70), height - product.height - int(height * 0.05))
    y = max(int(height * 0.08), y)

    shadow = Image.new("RGBA", target_size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_y = min(height - 1, y + product.height - int(product.height * 0.04))
    shadow_draw.ellipse(
        (
            max(0, x + int(product.width * 0.08)),
            max(0, shadow_y - int(height * 0.025)),
            min(width, x + int(product.width * 0.92)),
            min(height, shadow_y + int(height * 0.025)),
        ),
        fill=(30, 30, 30, 48),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(2, width // 90)))

    result = canvas.convert("RGBA")
    result.alpha_composite(shadow)
    result.alpha_composite(product, (x, y))
    return _with_flattened_data_alias(result.convert("RGB"))


def _with_flattened_data_alias(image: Image.Image) -> Image.Image:
    """兼容旧测试/调用方使用的 get_flattened_data 名称。"""

    if not hasattr(image, "get_flattened_data"):
        image.get_flattened_data = image.getdata  # type: ignore[attr-defined]
    return image


def _build_studio_background(
    target_size: tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT),
) -> Image.Image:
    """构造统一墙面和桌面的空场景底图。"""

    width, height = target_size
    canvas = Image.new("RGB", target_size, (236, 234, 229))
    draw = ImageDraw.Draw(canvas)

    # 桌面和墙面使用稳定的低对比度颜色，不抢商品主体。
    table_top = int(height * 0.72)
    draw.rectangle((0, table_top, width, height), fill=(216, 202, 183))
    draw.line((0, table_top, width, table_top), fill=(194, 180, 161), width=max(1, width // 360))
    return canvas
