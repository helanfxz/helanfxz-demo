from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any


_SPLIT_RE = re.compile(r"[,，、；;|\n]+")
_INTERNAL_INSTRUCTION_MARKERS = (
    "由 LLM 根据",
    "按 skill 选择",
    "根据所选表达策略",
    "具体剧情、动作和卖点证明方式交给 LLM",
)


def normalize_selling_points(points: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for point in points or []:
        for item in _SPLIT_RE.split(str(point)):
            item = item.strip()
            if item and item not in normalized:
                normalized.append(item)
    return normalized


def build_value_proof_plan(
    *,
    product_type: str,
    selling_points: Iterable[str],
    usage_scene: str = "",
    material_risk: str = "medium",
    asset_count: int = 1,
) -> dict[str, str]:
    """Fallback plan that is safe to pass to the video model.

    Normal production planning should still come from the LLM and skill
    examples. This fallback only prevents unresolved internal instructions from
    leaking into final prompts, and keeps the requested value tied to visible
    evidence rather than a fixed product choreography.
    """

    points = normalize_selling_points(selling_points)
    primary_value = points[0] if points else "卖点清楚"
    result_value = points[1] if len(points) > 1 else primary_value
    product = str(product_type or "商品").strip() or "商品"
    scene = str(usage_scene or "").strip()
    result_place = f"{scene}使用场景" if scene else "真实使用场景"
    result_evidence = _result_evidence_for_value(product=product, value=result_value, usage_scene=scene)

    return {
        "expression_strategy": "usage_result_demo",
        "primary_value": _short_caption(primary_value),
        "result_value": _short_caption(result_value),
        "confirm_caption": _short_caption(f"真实{product}"),
        "action_caption": _short_caption(primary_value),
        "result_caption": _short_caption(result_value),
        "source_place": "素材首帧场景",
        "result_place": result_place,
        "human": "普通用户",
        "source_action": _source_evidence_action(product=product, value=primary_value),
        "result_state": result_evidence["state"],
        "result_action": result_evidence["action"],
        "notes_for_review": "系统兜底只补齐可拍证据关系；具体创意优先使用 LLM 和 skill 样例输出。",
        "material_risk": str(material_risk or "medium"),
        "asset_count": str(asset_count),
    }


def ensure_value_proof_plan(
    plan: dict[str, Any],
    *,
    product_type: str,
    selling_points: Iterable[str],
    usage_scene: str = "",
    material_risk: str = "medium",
    asset_count: int = 1,
) -> dict[str, str]:
    """Remove unresolved instructions and repair obvious value-proof gaps."""

    fallback = build_value_proof_plan(
        product_type=product_type,
        selling_points=selling_points,
        usage_scene=usage_scene,
        material_risk=material_risk,
        asset_count=asset_count,
    )
    repaired: dict[str, str] = {key: str(value) for key, value in fallback.items()}
    for key, value in (plan or {}).items():
        if value is None:
            continue
        repaired[str(key)] = str(value).strip()

    primary_value = repaired.get("primary_value") or fallback["primary_value"]
    result_value = repaired.get("result_value") or fallback["result_value"]
    source_action = repaired.get("source_action", "")
    result_action = repaired.get("result_action", "")
    result_state = repaired.get("result_state", "")

    if _is_unresolved_instruction(source_action):
        repaired["source_action"] = _source_evidence_action(product=str(product_type or "商品"), value=primary_value)

    result_text = f"{result_state} {result_action}"
    if _is_unresolved_instruction(result_text) or not _result_text_proves_value(result_value, result_text):
        evidence = _result_evidence_for_value(
            product=str(product_type or "商品"),
            value=result_value,
            usage_scene=usage_scene,
        )
        repaired["result_state"] = evidence["state"]
        repaired["result_action"] = evidence["action"]
        notes = repaired.get("notes_for_review", "").strip()
        repair_note = f"已修正第三镜结果状态，使画面证据能证明「{_short_caption(result_value, 18)}」。"
        repaired["notes_for_review"] = f"{notes} {repair_note}".strip()

    for key in ("confirm_caption", "action_caption", "result_caption"):
        if _is_unresolved_instruction(repaired.get(key, "")):
            repaired[key] = fallback[key]
    return repaired


def _source_evidence_action(*, product: str, value: str) -> str:
    caption = _short_caption(value, 18)
    return (
        f"{product}保持在素材首帧场景中清楚可见，镜头只展示能画面证明「{caption}」的"
        "可见结构、材质、尺寸比例或与周边道具的关系，动作结束时停在稳定结果状态。"
    )


def _result_evidence_for_value(*, product: str, value: str, usage_scene: str) -> dict[str, str]:
    caption = _short_caption(value, 18)
    scene = str(usage_scene or "真实使用场景").strip()
    if _contains_any(caption, ("续航", "电池", "电量", "不插电")):
        return {
            "state": (
                f"{product}处在{scene}的不插电持续使用状态，桌面或周围没有充电器和电源线，"
                f"屏幕或工作状态持续可见，画面证明「{caption}」。"
            ),
            "action": "人物只做持续使用中的轻微敲键、查看或操作动作，不插电状态和无电源线环境始终清楚。",
        }
    if _contains_any(caption, ("多色", "颜色", "配色", "可选")):
        return {
            "state": (
                f"{product}在{scene}中以同一商品系列的颜色证据呈现，已有颜色差异、色卡、包装标识或多色并排关系清楚，"
                f"画面证明「{caption}」。"
            ),
            "action": "人物只做轻微指向或整理动作，让颜色差异或可选信息保持在画面主体位置。",
        }
    if _contains_any(caption, ("容量", "大杯", "装得", "收纳", "空间")):
        return {
            "state": (
                f"{product}在{scene}中和可容纳物、水位、包内空间或尺寸参照形成清楚比例关系，"
                f"画面证明「{caption}」。"
            ),
            "action": "人物只做围绕容量结果的轻微辅助动作，容量参照和商品主体始终清楚可见。",
        }
    if _contains_any(caption, ("便携", "轻薄", "随身", "通勤", "出门", "好收纳", "携带")):
        return {
            "state": (
                f"{product}在{scene}中和背包、工位、玄关或随身物品形成清楚携带关系，"
                f"画面证明「{caption}」。"
            ),
            "action": "人物只做轻微整理或继续使用动作，让携带关系、尺寸比例和商品主体保持清楚。",
        }
    if _contains_any(caption, ("保温", "保冷", "冷热", "温度")):
        return {
            "state": (
                f"{product}在{scene}中和热饮、冷饮、杯壁水汽或使用时长线索形成明确关系，"
                f"画面证明「{caption}」。"
            ),
            "action": "人物只做轻微使用动作，温度线索和商品主体始终清楚可见。",
        }
    if _contains_any(caption, ("性能", "流畅", "不卡", "效率", "高效")):
        return {
            "state": (
                f"{product}在{scene}中处于连续工作或高负载使用结果，屏幕、任务界面或操作反馈清楚，"
                f"画面证明「{caption}」。"
            ),
            "action": "人物只做持续操作动作，让运行状态和商品主体稳定可见。",
        }
    return {
        "state": f"{product}处在{scene}的真实使用结果状态，位置、接触关系和周边道具共同画面证明「{caption}」。",
        "action": "人物或道具只做围绕结果状态的轻微辅助动作，画面重点保持在商品和已成立的使用关系上。",
    }


def _result_text_proves_value(value: str, text: str) -> bool:
    caption = _short_caption(value, 18)
    if not caption:
        return True
    if _contains_any(caption, ("续航", "电池", "电量", "不插电")):
        return _contains_any(text, ("不插电", "电源线", "充电器", "电池", "电量", "持续使用", "一整天", "全天"))
    if _contains_any(caption, ("多色", "颜色", "配色", "可选")):
        return _contains_any(text, ("多色", "颜色", "配色", "色卡", "并排", "可选", "颜色差异"))
    if _contains_any(caption, ("容量", "大杯", "装得", "收纳", "空间")):
        return _contains_any(text, ("容量", "水位", "装满", "容纳", "包内空间", "尺寸", "比例", "参照"))
    if _contains_any(caption, ("便携", "轻薄", "随身", "通勤", "出门", "好收纳", "携带")):
        return _contains_any(text, ("背包", "随身", "携带", "通勤", "收纳", "尺寸", "比例", "工位", "玄关"))
    if _contains_any(caption, ("保温", "保冷", "冷热", "温度")):
        return _contains_any(text, ("热饮", "冷饮", "水汽", "冰块", "温度", "保温", "保冷", "冷热"))
    if _contains_any(caption, ("性能", "流畅", "不卡", "效率", "高效")):
        return _contains_any(text, ("运行", "任务", "操作", "屏幕", "流畅", "响应", "工作", "高负载"))
    return caption in text or "画面证明" in text or "结果状态" in text


def _is_unresolved_instruction(text: str) -> bool:
    return any(marker in str(text or "") for marker in _INTERNAL_INSTRUCTION_MARKERS)


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in str(text or "") for word in words)


def _short_caption(text: str, max_chars: int = 12) -> str:
    value = str(text or "").strip()
    return value[:max_chars] if value else "卖点清楚"
