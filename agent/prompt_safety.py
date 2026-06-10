"""Prompt safety helpers shared by video prompt builders.

The core contract is simple: product-free text-to-video shots must not carry
specific product, brand, logo, or appearance instructions into the video model.
"""

from __future__ import annotations

from typing import Any


def safe_text_to_video_scene_description(shot: dict[str, Any]) -> str:
    """Return a product-free scene description for text-to-video shots."""

    visual = str(shot.get("visual_description", "")).strip()
    initial = str(shot.get("initial_state", shot.get("scene_before", ""))).strip()
    final = str(shot.get("final_state", shot.get("scene_after", ""))).strip()
    action = str(shot.get("action", "")).strip()
    subject_appearance = str(shot.get("subject_appearance", "")).strip()
    subject_position = str(shot.get("subject_position", "")).strip()
    acting_direction = str(shot.get("acting_direction", "")).strip()
    scene_elements = shot.get("scene_elements", [])

    parts: list[str] = []
    if visual:
        parts.append(visual)
    elif initial:
        parts.append(initial)
        if action:
            parts.append(action)
        if final:
            parts.append(final)

    if subject_appearance:
        parts.append(f"人物外观：{subject_appearance}")
    if subject_position:
        parts.append(f"构图：{subject_position}")
    combined_so_far = visual + initial + action
    if acting_direction and acting_direction not in combined_so_far:
        parts.append(f"动作指导：{acting_direction}")
    if scene_elements:
        elems = scene_elements if isinstance(scene_elements, list) else [scene_elements]
        parts.append(f"画面元素：{'、'.join(str(e) for e in elems)}")

    if parts:
        base = "。".join(p.rstrip("。") for p in parts if p)
        if not scene_text_mentions_recognizable_product(base, shot):
            return base + product_free_scene_guard(shot)

    role = str(shot.get("narrative_role", "")).strip().lower()
    if role in {"hook", "problem"}:
        return "真实生活场景，人物在日常环境中自然活动，为后续商品展示留出空间。" + product_free_scene_guard(shot)
    return "真实生活场景，人物自然整理使用空间，只出现普通无品牌道具。" + product_free_scene_guard(shot)


def product_free_scene_guard(shot: dict[str, Any]) -> str:
    """Return guard text for shots where the branded product must not appear.

    Two responsibilities, both data-driven so they generalize to any product
    without product-type ``if/else`` branches:

    1. Forbid identity-bearing signals (a clear product subject, brand logo,
       brand text, readable label) so a text-to-video model can't invent a
       wrong-brand product.
    2. Positively KEEP a brandless, low-detail use-scene cue derived from the
       shot's own data, so the bridge shot still communicates what the upcoming
       product is for. Without this the shot reads as an unrelated clip once the
       subtitle is removed.
    """

    identity_card = shot.get("product_identity_card") or {}
    category = str(identity_card.get("product_type", "")).strip() or "该商品"

    guard = (
        "不出现待售商品或同类商品的清晰主体，不出现可识别的品牌 logo、品牌文字或可读标签；"
        f"凡是会让观众一眼认出具体品牌或型号的{category}都不要清晰展示。"
        "No clearly branded product, no logo, no brand text, no readable label."
    )

    cue = _product_free_context_cue(shot, category)
    if cue:
        guard += (
            f"但可以保留{cue}这类不带品牌、看不清商品细节的场景线索，"
            f"用来暗示接下来会用到{category}的使用场景，并和下一镜的真实商品自然衔接；"
            "这些线索只作环境铺垫，不能变成清晰的商品主体或露出任何标识。"
        )
    return guard


def _product_free_context_cue(shot: dict[str, Any], category: str) -> str:
    """Derive a brandless, identity-free context cue from the shot's own data.

    Reuses ``scene_elements`` / ``usage_scene`` already present on the shot, so
    it adapts to any product without product-type branches. Elements that name
    the product itself or a brand mark are dropped; the rest become positive
    bridge context. Falls back to a generic carrier/setting phrase interpolated
    from the category.
    """

    raw_elements = shot.get("scene_elements") or []
    if not isinstance(raw_elements, list):
        raw_elements = [raw_elements]

    category_low = category.lower()
    brand_markers = ("logo", "品牌", "标识", "商标", "字样")
    kept: list[str] = []
    for element in raw_elements:
        text = str(element).strip()
        if not text:
            continue
        low = text.lower()
        if category_low and category_low in low:
            continue
        if any(marker in low for marker in brand_markers):
            continue
        kept.append(text)

    if kept:
        return "、".join(dict.fromkeys(kept))

    usage_scene = str(shot.get("usage_scene", "")).strip()
    if usage_scene:
        return f"{usage_scene}里随身携带或摆放的无品牌物品"
    return f"携带或放置{category}的包袋、桌面位置或使用环境"


def scene_text_mentions_recognizable_product(text: str, shot: dict[str, Any]) -> bool:
    """Detect whether free text still asks the model to draw a concrete product."""

    normalized = text.lower()
    negated_markers = (
        "没有", "不出现", "不得出现", "禁止出现", "避免", "no ", "without", "must not", "do not",
    )
    has_negated_product_clause = any(marker in normalized for marker in negated_markers)
    product_keywords = (
        "product", "商品", "laptop", "notebook", "笔记本", "电脑", "水杯", "杯子",
        "鼠标", "键盘", "手机", "耳机",
    )
    brand_keywords = (
        "logo", "brand", "branded", "品牌", "商标", "标识", "雷蛇", "razer",
    )
    identity_card = shot.get("product_identity_card") or {}
    product_type = str(identity_card.get("product_type", "")).strip().lower()
    if product_type and product_type in normalized and not has_negated_product_clause:
        return True
    if has_negated_product_clause:
        return False
    return any(keyword in normalized for keyword in product_keywords) or any(
        keyword in normalized for keyword in brand_keywords
    )


def is_laptop_product(product_type: str) -> bool:
    """Recognize laptop products for hinge and prop ambiguity guards.

    Retained only for the renderer's hinge-stability guard; the product-free
    scene guard is now data-driven and no longer branches on product type.
    """

    normalized = product_type.lower()
    return any(keyword in normalized for keyword in ("笔记本", "laptop", "notebook"))
