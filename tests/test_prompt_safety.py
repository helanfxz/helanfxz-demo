from agent.prompt_safety import (
    product_free_scene_guard,
    safe_text_to_video_scene_description,
    scene_text_mentions_recognizable_product,
)


def test_product_forbidden_scene_replaces_brand_product_text():
    shot = {
        "render_strategy": "text_to_video",
        "product_presence": "forbidden",
        "visual_description": "一台雷蛇笔记本悬浮翻转",
        "scene_goal": "建立性能问题",
    }

    prompt = safe_text_to_video_scene_description(shot)

    assert "雷蛇笔记本" not in prompt
    assert "不出现待售商品或同类商品" in prompt


def test_product_free_guard_forbids_identity_and_keeps_brandless_bridge():
    guard = product_free_scene_guard(
        {
            "product_identity_card": {"product_type": "笔记本电脑"},
            "scene_elements": ["通勤包", "办公桌", "品牌 logo 贴纸"],
        }
    )

    # 通用：身份信号照禁，并按品类名插值，不靠 if/else 分支
    assert "品牌 logo" in guard
    assert "可读标签" in guard
    assert "笔记本电脑" in guard
    # 通用桥：保留无品牌场景线索，丢掉带品牌的元素
    assert "通勤包" in guard
    assert "办公桌" in guard
    assert "品牌 logo 贴纸" not in guard


def test_product_free_guard_generalizes_to_any_category():
    """同一套逻辑必须对非笔记本品类同样产出桥梁线索。"""

    guard = product_free_scene_guard(
        {
            "product_identity_card": {"product_type": "跑步鞋"},
            "scene_elements": ["玄关", "鞋柜"],
        }
    )

    assert "跑步鞋" in guard
    assert "玄关" in guard
    assert "鞋柜" in guard


def test_product_free_guard_falls_back_to_generic_carrier_cue():
    """没有可用场景元素时，回退到按品类插值的通用携带/使用情景。"""

    guard = product_free_scene_guard(
        {"product_identity_card": {"product_type": "保温杯"}, "scene_elements": []}
    )

    assert "保温杯" in guard
    assert "包袋" in guard or "使用环境" in guard


def test_scene_text_detects_brand_and_product_terms():
    assert scene_text_mentions_recognizable_product(
        "show a branded laptop and a large logo",
        {"product_identity_card": {"product_type": "laptop"}},
    )
