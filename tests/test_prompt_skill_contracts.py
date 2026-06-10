from agent.prompt_skill_contracts import (
    load_prompt_skill_contract,
    validate_prompt_skill_contract,
    validate_shot_skill_contract,
)


SKILLS = [
    "commerce_scene.source_confirm",
    "commerce_scene.material_action_proof",
    "commerce_scene.new_scene_result",
    "source_scene_extension.product_result_scene",
]


def test_commerce_prompt_skills_define_contract_metadata():
    for skill_id in SKILLS:
        contract = load_prompt_skill_contract(skill_id)
        issues = validate_prompt_skill_contract(contract)
        assert issues == [], f"{skill_id}: {issues}"


def test_missing_skill_contract_reports_path():
    contract = load_prompt_skill_contract("missing.family")
    assert contract["id"] == "missing.family"
    assert contract["exists"] is False
    assert "missing" in contract["path"]


def test_new_scene_result_cannot_be_forbidden_when_prompt_requests_product():
    shot = {
        "shot_index": 2,
        "selected_prompt_skill": "commerce_scene.new_scene_result",
        "render_strategy": "text_to_video",
        "product_presence": "forbidden",
        "visual_description": "同一件塑料吸管水杯已经在户外背包侧袋里，透明杯身清楚可见。",
    }

    issues = validate_shot_skill_contract(shot)

    assert any("product_presence=forbidden conflicts" in issue for issue in issues)


def test_source_scene_extension_cannot_mix_source_frame_and_new_location():
    shot = {
        "shot_index": 2,
        "selected_prompt_skill": "source_scene_extension.product_result_scene",
        "render_strategy": "image_to_video",
        "product_presence": "required",
        "asset_id": "asset-cup",
        "visual_description": "第一帧仍使用上传素材中的真实水杯，镜头一开始已经在户外公园长椅旁。",
    }

    issues = validate_shot_skill_contract(shot)

    assert any("source-frame skill requests a new scene" in issue for issue in issues)
