from agent.value_proof_planner import build_value_proof_plan, normalize_selling_points


def test_normalize_selling_points_splits_commas_and_chinese_punctuation():
    assert normalize_selling_points(["颜值在线，多色可选", "容量大", "冷热都能装"]) == [
        "颜值在线",
        "多色可选",
        "容量大",
        "冷热都能装",
    ]


def test_fallback_plan_is_neutral_and_skill_guided():
    plan = build_value_proof_plan(
        product_type="塑料吸管水杯",
        selling_points=["颜值在线，多色可选", "容量大", "冷热都能装"],
        usage_scene="通勤,户外",
        material_risk="low",
    )

    assert plan["primary_value"] == "颜值在线"
    assert plan["result_value"] == "多色可选"
    assert "由 LLM 根据" not in plan["source_action"]
    assert "由 LLM 根据" not in plan["result_action"]
    assert "skill" not in plan["source_action"]
    assert "skill" not in plan["result_action"]
    assert "画面证明" in plan["source_action"]
    assert "画面证明" in plan["result_state"]


def test_fallback_plan_does_not_contain_product_specific_choreography():
    plan = build_value_proof_plan(
        product_type="笔记本电脑",
        selling_points=["轻薄便携", "办公更顺手"],
        usage_scene="办公",
    )

    action_text = plan["source_action"] + plan["result_action"]
    assert "打开屏幕" not in action_text
    assert "触控板" not in action_text
    assert "平移笔记本" not in action_text


def test_fallback_plan_makes_battery_life_visually_provable():
    plan = build_value_proof_plan(
        product_type="笔记本电脑",
        selling_points=["高性能处理器", "长续航"],
        usage_scene="移动办公",
    )

    assert plan["result_value"] == "长续航"
    evidence_text = plan["result_state"] + plan["result_action"]
    assert "不插电" in evidence_text
    assert "电源线" in evidence_text or "充电器" in evidence_text
