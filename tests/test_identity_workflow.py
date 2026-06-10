import json
import os
from pathlib import Path

import agent.video_generation_workflow as workflow
import task_creation_demo_app as demo_app
import numpy as np
from PIL import Image
from agent.asset_preprocessor import (
    _compose_studio_anchor,
    _foreground_is_usable,
    _select_primary_foreground,
    create_studio_background,
)
from agent.seedance_video_renderer import (
    _adapt_clip_to_target_duration,
    _blend_transition_frames,
    _build_seedance_payload,
    _build_seedance_prompt,
    _build_subtitle_timeline,
    _concat_videos,
    _poll_seedance_task,
    _render_local_identity_anchor_clip,
    _render_seedance_batch,
    _render_local_scene_background_clip,
    repair_and_rerender_shot,
    _resolve_seedance_asset,
    _retime_video,
    _should_continue_from_previous,
    _should_render_scene_background_locally,
    _split_seedance_render_batches,
    _transition_types_for_clip_order,
    _public_upload_url_for_video,
)
from agent.simple_video_renderer import _build_storyboard_preview_text
from video_task_module import (
    CreateVideoTaskCommand,
    InMemoryTaskRepository,
    UploadedAsset,
    confirm_task_primary_product_selections,
    create_video_task,
    update_task_primary_product_preflight,
)
from agent.video_generation_workflow import (
    _apply_shared_anchor_fallback,
    _asset_analysis_for_llm,
    _build_content_repair_records,
    _clean_short_sentence,
    _enforce_storyboard_continuity_groups,
    _fallback_conservative_script,
    _fallback_conservative_storyboard,
    _fallback_director_storyboard,
    _save_workflow_artifacts,
    _normalize_script_plan,
    _normalize_storyboard,
    _plan_product_fidelity_v3_storyboard,
    _repair_rendered_content,
    _ensure_storyboard_continuity,
    _select_auto_repair_records,
    _shot_prefers_real_asset,
    adapt_storyboard_to_render_segments,
    build_creation_plan,
    build_product_context,
    build_product_identity_card,
    match_assets_to_storyboard,
    plan_storyboard_from_template,
    plan_director_storyboard,
    plan_script,
    repair_storyboard_by_shootability,
    review_storyboard_shootability,
    review_storyboard,
    review_script_plan,
    review_rendered_video_content,
    run_final_check,
)


def test_shared_anchor_fallback_keeps_original_for_analysis_but_reuses_clean_anchor_for_rendering():
    assets = [
        {
            "asset_id": "asset_detail",
            "asset_type": "image",
            "file_path": "/tmp/preprocessed_detail.jpg",
            "standardized_file_path": "/tmp/preprocessed_detail.jpg",
            "anchor_file_path": "",
        },
        {
            "asset_id": "asset_showcase",
            "asset_type": "image",
            "file_path": "/tmp/anchor_showcase.jpg",
            "standardized_file_path": "/tmp/preprocessed_showcase.jpg",
            "anchor_file_path": "/tmp/anchor_showcase.jpg",
            "visual_role": "appearance_anchor",
        },
    ]

    fallback_count = _apply_shared_anchor_fallback(assets)

    assert fallback_count == 1
    assert assets[0]["standardized_file_path"] == "/tmp/preprocessed_detail.jpg"
    assert assets[0]["file_path"] == "/tmp/anchor_showcase.jpg"
    assert assets[0]["render_anchor_source_asset_id"] == "asset_showcase"
    assert assets[0]["shared_anchor_fallback"] is True


def test_shared_anchor_fallback_does_not_promote_detail_reference_to_full_product_anchor():
    assets = [
        {
            "asset_id": "asset_logo",
            "asset_type": "image",
            "file_path": "/tmp/logo.jpg",
            "standardized_file_path": "/tmp/logo.jpg",
            "anchor_file_path": "/tmp/logo-anchor.jpg",
            "visual_role": "detail_reference",
        },
        {
            "asset_id": "asset_other",
            "asset_type": "image",
            "file_path": "/tmp/other.jpg",
            "standardized_file_path": "/tmp/other.jpg",
            "anchor_file_path": "",
            "visual_role": "detail_reference",
        },
    ]

    fallback_count = _apply_shared_anchor_fallback(assets)

    assert fallback_count == 0
    assert assets[1]["file_path"] == "/tmp/other.jpg"
    assert "shared_anchor_fallback" not in assets[1]


def test_create_studio_background_outputs_reusable_empty_scene(tmp_path):
    background_path = create_studio_background(str(tmp_path))

    background = Image.open(background_path)
    assert background.size == (720, 1280)
    assert all(abs(actual - expected) <= 2 for actual, expected in zip(background.getpixel((0, 0)), (236, 234, 229)))
    assert all(abs(actual - expected) <= 2 for actual, expected in zip(background.getpixel((0, 1279)), (216, 202, 183)))


def test_primary_foreground_selection_keeps_only_best_product_candidate(tmp_path):
    foreground = Image.new("RGBA", (240, 180), (0, 0, 0, 0))
    pixels = np.array(foreground)
    # 中心区域的大矩形是主商品，右侧小矩形模拟同一张图里的辅助商品。
    pixels[35:155, 45:145] = (180, 120, 80, 255)
    pixels[70:145, 175:225] = (60, 150, 90, 255)

    selected, profile = _select_primary_foreground(
        Image.fromarray(pixels),
        output_dir=str(tmp_path),
        stem="multi-product",
    )

    selected_alpha = np.array(selected.getchannel("A"))
    assert profile["candidate_count"] == 2
    assert profile["bbox"] == [45, 35, 145, 155]
    assert profile["selection_method"] == "automatic_area_and_center"
    assert profile["requires_user_confirmation"] is False
    assert selected_alpha[80, 80] == 255
    assert selected_alpha[100, 200] == 0
    assert Path(profile["mask_path"]).exists()


def test_primary_foreground_selection_requests_confirmation_when_candidates_are_close(tmp_path):
    foreground = Image.new("RGBA", (240, 180), (0, 0, 0, 0))
    pixels = np.array(foreground)
    pixels[45:145, 25:105] = (180, 120, 80, 255)
    pixels[45:145, 135:215] = (60, 150, 90, 255)

    _, profile = _select_primary_foreground(
        Image.fromarray(pixels),
        output_dir=str(tmp_path),
        stem="ambiguous-products",
    )

    assert profile["candidate_count"] == 2
    assert profile["requires_user_confirmation"] is True
    assert profile["confidence"] == 0.55


def test_process_assets_exposes_primary_product_selection_to_downstream(monkeypatch, tmp_path):
    image_path = tmp_path / "uploaded.jpg"
    Image.new("RGB", (32, 32), (255, 255, 255)).save(image_path)
    primary_product = {
        "bbox": [4, 5, 20, 24],
        "mask_path": str(tmp_path / "mask.png"),
        "confidence": 0.55,
        "candidate_count": 2,
        "requires_user_confirmation": True,
    }
    monkeypatch.setattr(
        workflow,
        "preprocess_all_assets",
        lambda assets, output_dir: [{
            "original_path": str(image_path),
            "output_path": str(image_path),
            "anchor_output_path": str(image_path),
            "primary_product": primary_product,
        }],
    )
    monkeypatch.setattr(
        workflow,
        "_call_multimodal_llm",
        lambda *args, **kwargs: {"ok": False, "content": "", "error": "skip llm in unit test"},
    )
    monkeypatch.setattr(workflow, "create_studio_background", lambda output_dir: "")

    result = workflow.process_assets({
        "task_id": "task-test",
        "title": "水杯",
        "uploaded_assets": [{
            "asset_id": "asset-1",
            "asset_type": "image",
            "file_path": str(image_path),
        }],
    })

    assert result["assets"][0]["primary_product"] == primary_product


def test_task_waits_for_primary_product_confirmation_when_preflight_is_ambiguous():
    repository = InMemoryTaskRepository()
    task = create_video_task(
        CreateVideoTaskCommand(
            title="双杯素材",
            selling_points=["轻便"],
            target_platform="tiktok",
            duration_seconds=15,
            style="product_showcase",
            uploaded_assets=[
                UploadedAsset(filename="cups.jpg", content_type="image/jpeg", file_path="/tmp/cups.jpg"),
            ],
        ),
        repository,
    )

    updated = update_task_primary_product_preflight(
        task.task_id,
        repository,
        {
            "/tmp/cups.jpg": {
                "candidate_count": 2,
                "requires_user_confirmation": True,
                "candidates": [{"bbox": [0, 0, 50, 90]}, {"bbox": [60, 0, 100, 90]}],
            },
        },
    )

    assert updated.status.value == "queued"
    assert updated.workflow_stage == "primary_product_confirmation"
    assert updated.uploaded_assets[0].primary_product["candidate_count"] == 2


def test_confirmed_primary_product_selection_is_saved_for_workflow():
    repository = InMemoryTaskRepository()
    task = create_video_task(
        CreateVideoTaskCommand(
            title="双杯素材",
            selling_points=["轻便"],
            target_platform="tiktok",
            duration_seconds=15,
            style="product_showcase",
            uploaded_assets=[
                UploadedAsset(filename="cups.jpg", content_type="image/jpeg", file_path="/tmp/cups.jpg"),
            ],
        ),
        repository,
    )
    update_task_primary_product_preflight(
        task.task_id,
        repository,
        {
            "/tmp/cups.jpg": {
                "candidate_count": 2,
                "requires_user_confirmation": True,
                "candidates": [{"bbox": [0, 0, 50, 90]}, {"bbox": [60, 0, 100, 90]}],
            },
        },
    )

    updated = confirm_task_primary_product_selections(task.task_id, repository, {0: 1})

    profile = updated.uploaded_assets[0].primary_product
    assert profile["selected_candidate_index"] == 1
    assert profile["selection_method"] == "user_confirmed"
    assert profile["requires_user_confirmation"] is False
    assert updated.workflow_stage == "created"


def test_primary_foreground_selection_uses_confirmed_candidate(tmp_path):
    foreground = Image.new("RGBA", (240, 180), (0, 0, 0, 0))
    pixels = np.array(foreground)
    pixels[35:155, 45:145] = (180, 120, 80, 255)
    pixels[70:145, 175:225] = (60, 150, 90, 255)

    selected, profile = _select_primary_foreground(
        Image.fromarray(pixels),
        output_dir=str(tmp_path),
        stem="confirmed-product",
        selected_candidate_index=1,
    )

    selected_alpha = np.array(selected.getchannel("A"))
    assert profile["selected_candidate_index"] == 1
    assert profile["selection_method"] == "user_confirmed"
    assert profile["requires_user_confirmation"] is False
    assert selected_alpha[80, 80] == 0
    assert selected_alpha[100, 200] == 255


def test_primary_product_confirmation_panel_renders_candidate_boxes():
    task = {
        "task_id": "task-confirm",
        "uploaded_assets": [{
            "filename": "cups.jpg",
            "public_url": "/uploads/task-confirm/01_cups.jpg",
            "primary_product": {
                "source_size": [200, 100],
                "requires_user_confirmation": True,
                "candidates": [
                    {"bbox": [10, 10, 80, 90], "score": 0.6},
                    {"bbox": [110, 20, 190, 95], "score": 0.58},
                ],
            },
        }],
    }

    html = demo_app._render_primary_product_confirmation(task)

    assert "/tasks/task-confirm/primary-product-confirmation" in html
    assert "selection_0_0" in html
    assert "selection_0_1" in html
    assert "请确认本次视频需要推广的主商品" in html


def test_primary_product_selection_form_values_are_parsed():
    selections = demo_app._parse_primary_product_selections(["0:1", "2:0"])

    assert selections == {0: 1, 2: 0}


def test_primary_product_confirmation_page_explains_that_workflow_is_waiting():
    html = demo_app._render_page(
        success_task={
            "task_id": "task-confirm",
            "status": "queued",
            "workflow_stage": "primary_product_confirmation",
            "workflow_message": "检测到多个相近商品，请确认本次视频需要推广的主商品。",
            "workflow_progress": 0,
            "workflow_events": [],
            "workflow_result": {},
            "title": "双杯素材",
            "target_platform": "tiktok",
            "duration_seconds": 15,
            "selling_points": ["轻便"],
            "uploaded_assets": [],
        },
        page_mode="detail",
    )

    assert "待确认主商品" in html
    assert "确认主商品后，系统才会启动工作流。" in html


def test_seedance_payload_requests_last_frame_and_accepts_previous_tail_as_first_frame():
    payload = _build_seedance_payload(
        model_endpoint="ep-video",
        shot={
            "render_strategy": "text_to_video",
            "purpose": "continue the desk scene",
            "visual_description": "same desk and same lighting",
        },
        first_frame_url_override="https://example.com/previous-last-frame.jpg",
    )

    assert payload["return_last_frame"] is True
    assert payload["content"][-1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/previous-last-frame.jpg"},
        "role": "first_frame",
    }


def test_seedance_payload_can_pull_continuation_back_to_real_asset_anchor(tmp_path):
    anchor_path = tmp_path / "anchor.jpg"
    anchor_path.write_bytes(b"fake-jpeg")
    payload = _build_seedance_payload(
        model_endpoint="ep-video",
        shot={
            "render_strategy": "image_to_video",
            "purpose": "keep product stable",
            "visual_description": "same product on same desk",
            "anchor_last_frame": True,
            "render_input": {
                "type": "asset",
                "asset_id": "asset-product",
                "asset_type": "image",
                "file_path": str(anchor_path),
            },
        },
        first_frame_url_override="https://example.com/previous-last-frame.jpg",
    )

    image_roles = [item["role"] for item in payload["content"] if item["type"] == "image_url"]
    assert image_roles == ["first_frame", "last_frame"]


def test_short_anchor_last_frame_segment_trims_instead_of_fast_forwarding(monkeypatch, tmp_path):
    source_path = tmp_path / "source.mp4"
    source_path.write_bytes(b"video")
    calls = []

    monkeypatch.setattr(
        "agent.seedance_video_renderer._retime_video",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not retime short anchored clip")),
    )
    monkeypatch.setattr(
        "agent.seedance_video_renderer._trim_video",
        lambda source_path, output_path, duration_seconds: (
            calls.append(("trim", duration_seconds))
            or {"success": True, "error": None}
        ),
    )

    result = _adapt_clip_to_target_duration(
        source_path=source_path,
        output_dir=tmp_path,
        shot_index=2,
        shot={"duration_seconds": 3, "anchor_last_frame": True},
    )

    assert result.name == "seedance_shot_02_trimmed.mp4"
    assert calls == [("trim", 3.0)]


def test_retime_video_keeps_concat_compatible_fps_and_exact_duration(monkeypatch, tmp_path):
    commands = []

    monkeypatch.setattr(
        "agent.seedance_video_renderer.subprocess.run",
        lambda command, **kwargs: (
            commands.append(command)
            or type("Completed", (), {"returncode": 0, "stderr": ""})()
        ),
    )

    result = _retime_video(
        source_path=tmp_path / "source.mp4",
        output_path=tmp_path / "retimed.mp4",
        duration_seconds=3.0,
        source_duration_seconds=5.0,
    )

    assert result["success"] is True
    assert "-r" in commands[0]
    assert commands[0][commands[0].index("-r") + 1] == "24"
    assert "-t" in commands[0]
    assert commands[0][commands[0].index("-t") + 1] == "3.000"


def test_poll_seedance_task_preserves_last_frame_url(monkeypatch):
    monkeypatch.setattr(
        "agent.seedance_video_renderer._send_json_request",
        lambda **kwargs: {
            "success": True,
            "data": {
                "status": "succeeded",
                "content": {
                    "video_url": "https://example.com/video.mp4",
                    "last_frame_url": "https://example.com/last-frame.jpg",
                },
            },
            "error": None,
        },
    )

    result = _poll_seedance_task("api-key", "task-id")

    assert result["success"] is True
    assert result["last_frame_url"] == "https://example.com/last-frame.jpg"


def test_only_explicit_same_group_shot_continues_from_previous_tail():
    previous = {"continuity_group": "desk_story", "transition_type": "hard_cut"}

    assert _should_continue_from_previous(
        {"continuity_group": "desk_story", "transition_type": "continue_from_previous"},
        previous,
    ) is True
    assert _should_continue_from_previous(
        {"continuity_group": "product_showcase", "transition_type": "continue_from_previous"},
        previous,
    ) is False
    assert _should_continue_from_previous(
        {"continuity_group": "desk_story", "transition_type": "hard_cut"},
        previous,
    ) is False


def test_transition_types_default_to_hard_cut_and_keep_explicit_crossfade():
    transitions = _transition_types_for_clip_order(
        shots_by_index={
            1: {"shot_index": 1},
            2: {"shot_index": 2},
            3: {"shot_index": 3, "transition_type": "crossfade"},
            4: {"shot_index": 4, "transition_type": "continue_from_previous"},
        },
        ordered_shot_indices=[1, 2, 3, 4],
    )

    assert transitions == ["hard_cut", "crossfade", "hard_cut"]


def test_seedance_batch_passes_previous_tail_to_explicit_continuation(monkeypatch):
    create_calls = []

    def fake_create(api_key, model_endpoint, shot, first_frame_url_override=""):
        create_calls.append(first_frame_url_override)
        return {"success": True, "task_id": f"task-{shot['shot_index']}", "error": None}

    def fake_poll(api_key, seedance_task_id):
        return {
            "success": True,
            "seedance_task_id": seedance_task_id,
            "status": "succeeded",
            "video_url": f"https://example.com/{seedance_task_id}.mp4",
            "last_frame_url": f"https://example.com/{seedance_task_id}-tail.jpg",
            "error": None,
        }

    monkeypatch.setattr("agent.seedance_video_renderer._create_seedance_task", fake_create)
    monkeypatch.setattr("agent.seedance_video_renderer._poll_seedance_task", fake_poll)

    results = _render_seedance_batch(
        api_key="api-key",
        model_endpoint="ep-video",
        indexed_shots=[
            (1, {"shot_index": 1, "continuity_group": "desk_story"}),
            (
                2,
                {
                    "shot_index": 2,
                    "continuity_group": "desk_story",
                    "transition_type": "continue_from_previous",
                },
            ),
        ],
    )

    assert [item["success"] for item in results] == [True, True]
    assert create_calls == ["", "https://example.com/task-1-tail.jpg"]


def test_concat_videos_uses_hard_cut_unless_crossfade_is_explicit(monkeypatch, tmp_path):
    calls = []
    clips = [tmp_path / "one.mp4", tmp_path / "two.mp4"]

    monkeypatch.setattr(
        "agent.seedance_video_renderer._concat_videos_without_transition",
        lambda clip_paths, final_video_path: calls.append("hard_cut") or {"success": True, "error": None},
    )
    monkeypatch.setattr(
        "agent.seedance_video_renderer._concat_videos_with_frame_crossfade",
        lambda clip_paths, final_video_path, transition_types=None: calls.append(transition_types) or {"success": True, "error": None},
    )

    _concat_videos(clips, tmp_path / "hard-cut.mp4")
    _concat_videos(clips, tmp_path / "crossfade.mp4", transition_types=["crossfade"])

    assert calls == ["hard_cut", ["crossfade"]]


def test_split_render_batches_only_groups_explicit_continuation_shots():
    batches = _split_seedance_render_batches(
        [
            (1, {"shot_index": 1, "continuity_group": "desk_story"}),
            (
                2,
                {
                    "shot_index": 2,
                    "continuity_group": "desk_story",
                    "transition_type": "continue_from_previous",
                },
            ),
            (3, {"shot_index": 3, "continuity_group": "product_showcase"}),
        ]
    )

    assert [[shot_index for shot_index, _ in batch] for batch in batches] == [[1, 2], [3]]


def test_seedance_prompt_inherits_visual_style_bible():
    prompt = _build_seedance_prompt(
        {
            "render_strategy": "text_to_video",
            "purpose": "establish commuter context",
            "visual_description": "desk beside a commuter bag",
            "product_presence": "forbidden",
            "visual_style_bible": {
                "realism": "photorealistic commercial video",
                "lighting": "soft daylight from camera left",
                "color_temperature": "neutral warm",
                "background_complexity": "clean and restrained",
            },
        }
    )

    assert "soft daylight from camera left" in prompt
    assert "neutral warm" in prompt


def test_seedance_template_prompt_still_gets_structured_identity_constraints():
    prompt = _build_seedance_prompt(
        {
            "render_strategy": "image_to_video",
            "seedance_prompt": "画面主体：同一只保温水杯稳定摆放在办公桌上，杯身清楚可见，人物手部轻轻扶住杯身。",
            "product_presence": "required",
            "identity_strictness": "high",
            "visual_style_bible": {
                "realism": "真实写实的商业短视频",
                "lighting": "柔和自然光，主体照明稳定",
                "color_temperature": "中性偏暖色温",
                "background_complexity": "背景克制干净",
                "camera_language": "稳定镜头，运动幅度小",
            },
            "product_identity_card": {
                "product_type": "保温水杯",
                "appearance_summary": "哑光黑色不锈钢杯身",
                "must_preserve": ["杯身轮廓", "黑色磨砂材质"],
            },
            "render_input": {
                "type": "asset",
                "asset_id": "cup_anchor",
                "file_path": "/tmp/cup_anchor.png",
                "asset_type": "image",
            },
        }
    )

    assert "图生视频，使用上传素材作为首帧" in prompt
    assert "后续画面必须延续首帧中的同一件商品" in prompt
    assert "真实写实的商业短视频" in prompt
    assert "杯身轮廓" in prompt


def test_template_storyboard_mixes_real_product_anchors_with_product_free_story_scene(monkeypatch):
    monkeypatch.setattr(
        "agent.video_generation_workflow._call_text_llm",
        lambda *args, **kwargs: {"ok": False, "content": "", "error": "disabled"},
    )

    storyboard, script_plan = plan_storyboard_from_template(
        {
            "product_identity_card": {
                "brand_name": "YETI",
                "product_type": "保温水杯",
                "appearance_summary": "哑光黑色不锈钢杯身，侧面有白色标识",
                "visible_marks": ["白色标识"],
                "texture_notes": "磨砂不锈钢表面",
                "identity_confidence": "high",
            },
            "target_audience": "通勤上班族",
            "usage_scene": "通勤、办公",
            "selling_points": ["大容量一杯管一天", "防漏设计随手包里放", "保温持久"],
            "visual_style_bible": {"realism": "真实写实"},
        },
        {
            "asset_profiles": [
                {
                    "asset_id": "cup_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/cup_anchor.png",
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                }
            ],
            "assets": [
                {
                    "asset_id": "cup_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/cup_anchor.png",
                    "is_supported": True,
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                }
            ],
        },
    )

    assert script_plan["_source"] == "product_fidelity_v3_skill_guided"
    assert [shot["narrative_role"] for shot in storyboard] == [
        "product_reveal",
        "feature_demo",
        "commerce_result",
    ]
    assert [shot["duration_seconds"] for shot in storyboard] == [5, 5, 5]
    opening = storyboard[0]
    assert opening["render_strategy"] == "image_to_video"
    assert opening["product_presence"] == "required"
    assert opening["identity_strictness"] == "high"
    assert opening["asset_id"] == "cup_anchor"
    assert opening["material_strategy"] == "source_scene_extension"
    assert opening["selected_prompt_skill"] == "commerce_scene.source_confirm"
    assert "拿起" not in opening["action"]

    for shot in storyboard[:2]:
        assert shot["render_strategy"] == "image_to_video"
        assert shot["product_presence"] == "required"
        assert shot["identity_strictness"] == "high"
        assert shot["asset_id"] == "cup_anchor"
        assert shot["transition_type"] == "hard_cut"
    assert storyboard[2]["render_strategy"] == "image_to_video"
    assert storyboard[2]["product_presence"] == "required"
    assert storyboard[2]["asset_id"] == "cup_anchor"
    assert "不是无商品铺垫镜头" in storyboard[2]["visual_description"]
    assert "source_scene_extension.product_result_scene" == storyboard[2]["selected_prompt_skill"]
    assert "不要新增非商品自带文字" in storyboard[2]["visual_description"]
    assert not any("喝水" in shot["action"] or "嘴边" in shot["action"] for shot in storyboard)


def test_product_fidelity_v3_laptop_uses_verified_dynamic_templates(monkeypatch):
    monkeypatch.setattr(
        "agent.video_generation_workflow._call_text_llm",
        lambda *args, **kwargs: {"ok": False, "content": "", "error": "disabled"},
    )

    storyboard, script_plan = plan_storyboard_from_template(
        {
            "product_identity_card": {
                "brand_name": "Razer",
                "product_type": "笔记本电脑",
                "appearance_summary": "黑色磨砂金属雷蛇笔记本，闭合状态，A 面中央有绿色三头蛇标识",
                "visible_marks": ["绿色三头蛇标识"],
                "identity_confidence": "high",
            },
            "selling_points": ["轻薄机身", "通勤好收纳"],
            "usage_scene": "通勤办公",
        },
        {
            "asset_profiles": [
                {
                    "asset_id": "laptop_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/laptop_anchor.png",
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                }
            ],
            "assets": [
                {
                    "asset_id": "laptop_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/laptop_anchor.png",
                    "is_supported": True,
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                }
            ],
        },
    )

    assert script_plan["_source"] == "product_fidelity_v3_skill_guided"
    assert [shot["render_strategy"] for shot in storyboard] == [
        "image_to_video",
        "image_to_video",
        "image_to_video",
    ]
    assert [shot["duration_seconds"] for shot in storyboard] == [5, 5, 5]
    assert all(shot["transition_type"] == "hard_cut" for shot in storyboard)
    assert storyboard[0]["product_presence"] == "required"
    assert storyboard[2]["product_presence"] == "required"
    assert storyboard[2]["narrative_role"] == "commerce_result"
    assert storyboard[0]["material_strategy"] == "source_scene_extension"
    assert storyboard[0]["selected_prompt_skill"] == "commerce_scene.source_confirm"
    actions = " ".join(shot["action"] for shot in storyboard)
    assert "skill" in actions or "素材" in actions
    assert "不拿起整件商品" in actions or "稳定" in actions
    assert "再把商品短距离拿起" not in actions
    assert "旋转" not in actions
    assert "跨面" not in actions
    assert "不得变成其他商品" in " ".join(storyboard[0]["forbidden_variation"])


def test_product_fidelity_v3_does_not_emit_internal_action_placeholders(monkeypatch):
    monkeypatch.setattr(
        "agent.video_generation_workflow._call_text_llm",
        lambda *args, **kwargs: {"ok": False, "content": "", "error": "disabled"},
    )

    storyboard, _script_plan = plan_storyboard_from_template(
        {
            "product_identity_card": {
                "product_type": "水杯",
                "appearance_summary": "透明棕色杯身，蓝色杯环，黄色杯盖",
                "identity_confidence": "high",
            },
            "selling_points": ["通勤随手带", "容量大", "冷热都能装"],
            "usage_scene": "通勤、办公室",
        },
        {
            "asset_profiles": [
                {
                    "asset_id": "cup_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/cup_anchor.png",
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                }
            ],
            "assets": [
                {
                    "asset_id": "cup_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/cup_anchor.png",
                    "is_supported": True,
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                }
            ],
        },
    )

    prompt_text = "\n".join(
        str(shot.get(field, ""))
        for shot in storyboard
        for field in ("action", "visual_description", "video_prompt")
    )
    assert "按 skill 选择" not in prompt_text
    assert "由 LLM 根据" not in prompt_text
    assert "具体剧情、动作和卖点证明方式交给 LLM" not in prompt_text
    assert "画面证明" in prompt_text


def test_product_fidelity_v3_uses_assets_path_when_profile_lacks_file_path(monkeypatch):
    monkeypatch.setattr(
        "agent.video_generation_workflow._call_text_llm",
        lambda *args, **kwargs: {"ok": False, "content": "", "error": "disabled"},
    )

    storyboard, script_plan = plan_storyboard_from_template(
        {
            "product_identity_card": {
                "product_type": "水杯",
                "appearance_summary": "透明棕色水杯，蓝色环和黄色杯盖",
                "identity_confidence": "low",
                "appearance_anchor_available": True,
            },
            "selling_points": ["通勤随手带"],
        },
        {
            "asset_profiles": [
                {
                    "asset_id": "cup_anchor",
                    "asset_type": "image",
                    "visual_role": "appearance_anchor",
                    "quality_score": 80,
                }
            ],
            "assets": [
                {
                    "asset_id": "cup_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/cup_anchor.png",
                    "anchor_file_path": "/tmp/cup_anchor.png",
                    "is_supported": True,
                    "visual_role": "appearance_anchor",
                    "quality_score": 80,
                }
            ],
        },
    )

    assert script_plan["_source"] == "product_fidelity_v3_skill_guided"
    assert [shot["render_strategy"] for shot in storyboard] == ["image_to_video", "image_to_video", "image_to_video"]
    assert storyboard[0]["asset_id"] == "cup_anchor"
    assert storyboard[1]["asset_id"] == "cup_anchor"
    assert storyboard[2]["asset_id"] == "cup_anchor"
    assert storyboard[0]["material_strategy"] == "source_scene_extension"


def test_product_fidelity_v3_uses_detail_reference_for_feature_shot():
    storyboard = _plan_product_fidelity_v3_storyboard(
        {
            "product_identity_card": {
                "product_type": "笔记本电脑",
                "appearance_summary": "黑色闭合笔记本，A 面有绿色标识",
            },
            "selling_points": ["轻薄机身，随手带走", "接口细节清楚可见"],
        },
        {
            "asset_profiles": [
                {
                    "asset_id": "asset_detail",
                    "visual_role": "detail_reference",
                    "suitable_for": ["detail_closeup", "feature_detail"],
                    "quality_score": 95,
                    "reason": "局部接口和标识清楚，适合细节证明。",
                },
                {
                    "asset_id": "asset_anchor",
                    "visual_role": "appearance_anchor",
                    "suitable_for": ["product_showcase"],
                    "quality_score": 90,
                    "reason": "整机完整，适合建立商品身份。",
                },
            ],
            "assets": [
                {
                    "asset_id": "asset_detail",
                    "asset_type": "image",
                    "file_path": "/tmp/detail.png",
                    "is_supported": True,
                    "visual_role": "detail_reference",
                    "quality_score": 95,
                },
                {
                    "asset_id": "asset_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/anchor.png",
                    "is_supported": True,
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                },
            ],
        },
        {
            "asset_id": "asset_anchor",
            "asset_type": "image",
            "file_path": "/tmp/anchor.png",
            "visual_role": "appearance_anchor",
            "quality_score": 90,
        },
    )

    assert storyboard[0]["asset_id"] == "asset_anchor"
    assert storyboard[0]["material_strategy"] == "source_scene_extension"
    assert storyboard[0]["asset_usage"]["visual_role"] == "appearance_anchor"
    feature = storyboard[1]
    assert feature["asset_id"] == "asset_detail"
    assert feature["material_strategy"] == "detail_reference"
    assert feature["asset_usage"]["visual_role"] == "detail_reference"
    assert "细节证明" in feature["asset_usage_reason"]


def test_v3_subtitle_cleaning_does_not_hard_cut_at_comma():
    text = _clean_short_sentence("高性能处理器，流畅不卡顿，精致做工，流畅不卡顿", max_chars=12)

    assert text == "高性能处理器，流畅不卡顿"
    assert not text.endswith("，")
    assert "精致做工" not in text


def test_product_fidelity_v3_product_shots_keep_hard_cut_reanchors():
    storyboard = _enforce_storyboard_continuity_groups(
        [
            {
                "shot_index": 0,
                "narrative_role": "hook",
                "product_presence": "required",
                "render_strategy": "image_to_video",
                "transition_type": "hard_cut",
                "planner_source": "product_fidelity_v3_skill_guided",
            },
            {
                "shot_index": 1,
                "narrative_role": "feature_demo",
                "product_presence": "required",
                "render_strategy": "image_to_video",
                "transition_type": "hard_cut",
                "planner_source": "product_fidelity_v3_skill_guided",
            },
        ]
    )

    assert [shot["transition_type"] for shot in storyboard] == ["hard_cut", "hard_cut"]
    assert [shot["continuity_group"] for shot in storyboard] == ["", ""]
    assert [shot["anchor_last_frame"] for shot in storyboard] == [False, False]


def test_product_fidelity_v3_verified_hand_action_passes_shootability_without_usage_asset():
    storyboard, _ = plan_storyboard_from_template(
        {
            "product_identity_card": {
                "product_type": "笔记本电脑",
                "appearance_summary": "黑色磨砂金属雷蛇笔记本，闭合状态，A 面中央有绿色三头蛇标识",
                "must_preserve": ["黑色机身", "绿色三头蛇标识"],
            },
            "selling_points": ["轻薄机身", "通勤好收纳"],
        },
        {
            "asset_profiles": [
                {
                    "asset_id": "laptop_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/laptop_anchor.png",
                    "visual_role": "appearance_anchor",
                }
            ],
            "assets": [
                {
                    "asset_id": "laptop_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/laptop_anchor.png",
                    "is_supported": True,
                    "visual_role": "appearance_anchor",
                }
            ],
        },
    )

    review = review_storyboard_shootability(
        storyboard,
        {"asset_capability_plan": {"appearance_anchor_available": True}},
        {
            "assets": [
                {
                    "asset_id": "laptop_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/laptop_anchor.png",
                    "is_supported": True,
                }
            ]
        },
    )
    repaired = repair_storyboard_by_shootability(
        storyboard,
        review,
        {"asset_capability_plan": {"appearance_anchor_available": True}},
        {"assets": []},
    )

    assert review["passed"] is True
    assert any("画面证明" in shot["action"] or "素材" in shot["action"] for shot in repaired)


def test_template_path_without_product_anchor_uses_director_planner(monkeypatch):
    calls = []

    def fake_plan_script(product_context):
        calls.append(("script", product_context["product_type"]))
        return {
            "hook": "通勤前的收纳问题",
            "body": ["放进包里不占空间", "随手拿取更方便"],
            "cta": "出门携带更省心",
            "beats": [],
        }

    def fake_director_storyboard(product_context, script_plan, asset_analysis):
        calls.append(("director", script_plan["hook"], len(asset_analysis.get("assets", []))))
        return [
            {
                "shot_index": 0,
                "narrative_role": "hook",
                "duration_seconds": 3,
                "render_strategy": "text_to_video",
                "product_presence": "optional",
                "identity_strictness": "low",
                "seedance_prompt": "用户出门前整理通勤包，画面呈现包内空间紧张的问题。",
                "subtitle": "通勤收纳别将就",
            },
            {
                "shot_index": 1,
                "narrative_role": "feature_demo",
                "duration_seconds": 4,
                "render_strategy": "text_to_video",
                "product_presence": "optional",
                "identity_strictness": "low",
                "seedance_prompt": "用户把水杯放进包侧袋，背包后自然走出家门。",
                "subtitle": "放进包里不占空间",
            },
            {
                "shot_index": 2,
                "narrative_role": "detail_proof",
                "duration_seconds": 4,
                "render_strategy": "text_to_video",
                "product_presence": "optional",
                "identity_strictness": "low",
                "seedance_prompt": "用户在地铁站外从包侧袋拿出水杯，动作自然连贯。",
                "subtitle": "随手拿取更方便",
            },
            {
                "shot_index": 3,
                "narrative_role": "cta",
                "duration_seconds": 3,
                "render_strategy": "text_to_video",
                "product_presence": "optional",
                "identity_strictness": "low",
                "seedance_prompt": "用户背包步行离开，水杯稳定收纳在侧袋中。",
                "subtitle": "出门携带更省心",
            },
        ]

    monkeypatch.setattr(workflow, "plan_script", fake_plan_script)
    monkeypatch.setattr(workflow, "plan_director_storyboard", fake_director_storyboard)

    storyboard, script_plan = workflow.plan_storyboard_from_template(
        {
            "product_type": "水杯",
            "duration_seconds": 15,
            "selling_points": ["放进包里不占空间", "随手拿取更方便"],
            "product_identity_card": {"product_type": "水杯"},
        },
        {"assets": [], "asset_profiles": []},
    )

    assert calls == [("script", "水杯"), ("director", "通勤前的收纳问题", 0)]
    assert script_plan["_source"] == "template_path_b_no_anchor_director"
    assert [shot["narrative_role"] for shot in storyboard] == [
        "hook",
        "feature_demo",
        "detail_proof",
        "cta",
    ]


def test_legacy_logo_closeup_templates_removed_from_runtime_module():
    source = Path(workflow.__file__).read_text(encoding="utf-8")

    removed_symbols = [
        "_SHOT_TEMPLATE",
        "_SHOT_TEMPLATE_IDENTITY_ANCHORED",
        "_template_fallback_storyboard",
        "_template_identity_anchored_fallback_storyboard",
        "_finalize_template_storyboard_continuity",
    ]
    for symbol in removed_symbols:
        assert symbol not in source

    removed_phrases = [
        "logo特写",
        "极致特写",
        "画面被logo和材质表面完全填充",
        "焦点精确在logo",
        "品牌标识必须精确保持",
    ]
    for phrase in removed_phrases:
        assert phrase not in source


def test_match_assets_keeps_story_context_scene_as_text_to_video():
    matches = match_assets_to_storyboard(
        storyboard=[
            {
                "shot_index": 1,
                "narrative_role": "problem",
                "product_presence": "forbidden",
                "render_strategy": "text_to_video",
            }
        ],
        asset_analysis={
            "assets": [],
            "shared_scene_background_path": "/tmp/studio_background.jpg",
        },
    )

    match = matches[0]
    assert match["strategy"] == "text_to_video"
    assert match["matched_asset"] is None
    assert match["render_input"] is None


def test_match_assets_binds_shared_scene_background_only_for_product_reveal():
    matches = match_assets_to_storyboard(
        storyboard=[
            {
                "shot_index": 1,
                "narrative_role": "product_reveal",
                "product_presence": "optional",
                "render_strategy": "image_to_video",
            }
        ],
        asset_analysis={
            "assets": [],
            "shared_scene_background_path": "/tmp/studio_background.jpg",
        },
    )

    match = matches[0]
    assert match["strategy"] == "image_to_video"
    assert match["matched_asset"]["is_scene_background"] is True
    assert match["render_input"]["file_path"] == "/tmp/studio_background.jpg"


def test_product_reveal_prefers_explicit_real_asset_over_shared_scene_background():
    matches = match_assets_to_storyboard(
        storyboard=[
            {
                "shot_index": 1,
                "narrative_role": "product_reveal",
                "continuity_mode": "shared_scene_bridge",
                "product_presence": "required",
                "render_strategy": "image_to_video",
                "asset_id": "asset-real",
            }
        ],
        asset_analysis={
            "shared_scene_background_path": "/tmp/studio_background.jpg",
            "assets": [
                {
                    "asset_id": "asset-real",
                    "asset_type": "image",
                    "file_path": "/tmp/anchor.jpg",
                    "anchor_file_path": "/tmp/anchor.jpg",
                    "visual_role": "appearance_anchor",
                    "is_supported": True,
                }
            ],
        },
    )

    match = matches[0]
    assert match["matched_asset"]["asset_id"] == "asset-real"
    assert match["matched_asset"].get("is_scene_background") is not True
    assert match["render_input"]["file_path"] == "/tmp/anchor.jpg"


def test_product_reveal_prefers_complete_hero_variant_even_when_review_mentions_logo(tmp_path):
    hero_path = tmp_path / "hero.jpg"
    detail_path = tmp_path / "detail.jpg"
    hero_path.write_bytes(b"hero")
    detail_path.write_bytes(b"detail")

    render_asset = workflow._select_render_asset_variant(
        {
            "narrative_role": "product_reveal",
            "scene_goal": "展示商品完整外观和 logo",
            "review_focus": ["检查品牌标识是否稳定"],
        },
        {
            "asset_id": "asset-real",
            "file_path": str(hero_path),
            "keyframe_variants": {
                "hero": str(hero_path),
                "detail": str(detail_path),
            },
        },
    )

    assert render_asset["keyframe_variant"] == "hero"
    assert render_asset["file_path"] == str(hero_path)


def test_product_reveal_replaces_explicit_detail_reference_with_complete_appearance_anchor():
    matches = match_assets_to_storyboard(
        storyboard=[
            {
                "shot_index": 1,
                "narrative_role": "product_reveal",
                "product_presence": "required",
                "render_strategy": "image_to_video",
                "asset_id": "asset-logo-detail",
            }
        ],
        asset_analysis={
            "assets": [
                {
                    "asset_id": "asset-logo-detail",
                    "asset_type": "image",
                    "file_path": "/tmp/logo-detail.jpg",
                    "visual_role": "detail_reference",
                    "quality_score": 100,
                    "is_supported": True,
                },
                {
                    "asset_id": "asset-full-product",
                    "asset_type": "image",
                    "file_path": "/tmp/full-product.jpg",
                    "visual_role": "appearance_anchor",
                    "quality_score": 80,
                    "is_supported": True,
                },
            ],
        },
    )

    assert matches[0]["matched_asset"]["asset_id"] == "asset-full-product"


def test_product_reveal_marks_detail_only_binding_as_creative_completion():
    matches = match_assets_to_storyboard(
        storyboard=[
            {
                "shot_index": 1,
                "narrative_role": "product_reveal",
                "product_presence": "required",
                "render_strategy": "image_to_video",
                "asset_id": "asset-logo-detail",
            }
        ],
        asset_analysis={
            "assets": [
                {
                    "asset_id": "asset-logo-detail",
                    "asset_type": "image",
                    "file_path": "/tmp/logo-detail.jpg",
                    "visual_role": "detail_reference",
                    "quality_score": 100,
                    "is_supported": True,
                }
            ],
        },
    )

    assert matches[0]["matched_asset"]["asset_id"] == "asset-logo-detail"
    assert matches[0]["reference_scope"] == "detail"
    assert matches[0]["creative_completion_required"] is True


def test_cta_uses_complete_hero_variant_instead_of_shrunken_cta_variant(tmp_path):
    hero_path = tmp_path / "hero.jpg"
    cta_path = tmp_path / "cta.jpg"
    hero_path.write_bytes(b"hero")
    cta_path.write_bytes(b"cta")

    render_asset = workflow._select_render_asset_variant(
        {
            "narrative_role": "cta",
            "product_presence": "required",
        },
        {
            "asset_id": "asset-full-product",
            "visual_role": "appearance_anchor",
            "file_path": str(hero_path),
            "keyframe_variants": {
                "hero": str(hero_path),
                "cta": str(cta_path),
            },
        },
    )

    assert render_asset["keyframe_variant"] == "hero"
    assert render_asset["file_path"] == str(hero_path)


def test_cta_binds_complete_product_anchor_even_when_storyboard_requests_text_to_video():
    matches = match_assets_to_storyboard(
        storyboard=[
            {
                "shot_index": 1,
                "narrative_role": "cta",
                "product_presence": "optional",
                "render_strategy": "text_to_video",
            }
        ],
        asset_analysis={
            "assets": [
                {
                    "asset_id": "asset-full-product",
                    "asset_type": "image",
                    "file_path": "/tmp/full-product.jpg",
                    "visual_role": "appearance_anchor",
                    "quality_score": 80,
                    "is_supported": True,
                }
            ],
        },
    )

    assert matches[0]["strategy"] == "image_to_video"
    assert matches[0]["matched_asset"]["asset_id"] == "asset-full-product"
    assert matches[0]["reference_scope"] == "full_product"


def test_creation_plan_enables_identity_tail_anchor_for_complete_product_shot():
    plan = build_creation_plan(
        product_context={"product_identity_card": {}},
        storyboard=[
            {
                "shot_index": 1,
                "duration_seconds": 3,
                "narrative_role": "cta",
                "visual_description": "show the real laptop",
                "subtitle": "learn more",
                "voiceover": "learn more",
            }
        ],
        asset_matching=[
            {
                "shot_index": 1,
                "strategy": "image_to_video",
                "reference_scope": "full_product",
                "matched_asset": {
                    "asset_id": "asset-full-product",
                    "asset_type": "image",
                    "file_path": "/tmp/full-product.jpg",
                },
            }
        ],
    )

    assert plan["shots"][0]["preserve_identity_tail"] is True


def test_seedance_payload_uses_same_real_asset_as_first_and_last_identity_anchor(tmp_path):
    anchor_path = tmp_path / "anchor.jpg"
    anchor_path.write_bytes(b"fake-jpeg")
    payload = _build_seedance_payload(
        model_endpoint="ep-video",
        shot={
            "render_strategy": "image_to_video",
            "preserve_identity_tail": True,
            "render_input": {
                "type": "asset",
                "asset_id": "asset-product",
                "asset_type": "image",
                "file_path": str(anchor_path),
            },
        },
    )

    image_roles = [item["role"] for item in payload["content"] if item["type"] == "image_url"]
    assert image_roles == ["first_frame", "last_frame"]
    assert payload["content"][1]["image_url"]["url"] == payload["content"][2]["image_url"]["url"]


def test_text_to_video_without_asset_uses_product_free_scene_description():
    prompt = _build_seedance_prompt(
        {
            "render_strategy": "text_to_video",
            "product_presence": "optional",
            "visual_description": "show a branded laptop and a large logo on the screen",
            "action": "rotate the laptop",
            "narrative_role": "hook",
            "product_identity_card": {"product_type": "laptop"},
        }
    )

    assert "show a branded laptop" not in prompt
    assert "rotate the laptop" not in prompt


def test_brand_logo_review_uses_deterministic_local_identity_fallback():
    records = _build_content_repair_records(
        [
            {
                "shot_index": 2,
                "pass": False,
                "failed_dimensions": ["brand_or_logo_consistency"],
                "main_issue": "logo drifted",
                "repair_strategy": "rerender_with_stronger_identity_anchor",
            }
        ]
    )

    assert records[0]["action"] == "fallback_to_local_identity_anchor"
    assert records[0]["repair_strategy"] == "fallback_to_local_identity_anchor"


def test_auto_content_repair_skips_systemic_failures():
    records = [{"shot_index": index, "repair_strategy": "simplify_action"} for index in range(1, 6)]

    selected, policy = _select_auto_repair_records(records, failed_count=5, total_shots=5)

    assert selected == []
    assert policy["auto_repair_enabled"] is False
    assert policy["selected_count"] == 0
    assert policy["original_count"] == 5
    assert "系统性计划失败" in policy["reason"]


def test_auto_content_repair_limits_local_failures_to_two_records():
    records = [{"shot_index": index, "repair_strategy": "simplify_action"} for index in range(1, 5)]

    selected, policy = _select_auto_repair_records(records, failed_count=2, total_shots=5)

    assert [record["shot_index"] for record in selected] == [1, 2]
    assert policy["auto_repair_enabled"] is True
    assert policy["selected_count"] == 2


def test_local_identity_anchor_clip_keeps_uploaded_asset_pixels(tmp_path):
    anchor_path = tmp_path / "anchor.jpg"
    output_path = tmp_path / "identity.mp4"
    Image.new("RGB", (720, 1280), (12, 120, 210)).save(anchor_path)

    result = _render_local_identity_anchor_clip(
        {
            "duration_seconds": 1,
            "render_input": {
                "type": "asset",
                "asset_id": "asset-product",
                "asset_type": "image",
                "file_path": str(anchor_path),
            },
        },
        output_path,
    )

    assert result["success"] is True
    assert output_path.exists()


def test_content_repair_reconcat_keeps_local_scene_clip(monkeypatch, tmp_path):
    (tmp_path / "seedance_shot_01.mp4").write_bytes(b"old")
    (tmp_path / "seedance_shot_02_local_scene.mp4").write_bytes(b"scene")
    repaired_clip = tmp_path / "seedance_shot_01_local_identity.mp4"
    repaired_clip.write_bytes(b"repaired")
    concat_calls = []

    monkeypatch.setattr(
        workflow,
        "repair_and_rerender_shot",
        lambda **kwargs: {"success": True, "clip_path": str(repaired_clip)},
    )
    monkeypatch.setattr(
        "agent.seedance_video_renderer._concat_videos",
        lambda clip_paths, final_video_path, transition_types=None: (
            concat_calls.append([path.name for path in clip_paths])
            or {"success": True, "error": None}
        ),
    )
    monkeypatch.setattr(
        "agent.seedance_video_renderer._overlay_storyboard_subtitles",
        lambda **kwargs: {"success": True, "error": None},
    )

    result = _repair_rendered_content(
        task_id="task-1",
        repair_records=[{"shot_index": 1, "repair_strategy": "fallback_to_local_identity_anchor"}],
        creation_plan={"shots": [{"shot_index": 1}, {"shot_index": 2}]},
        render_result={"shot_results": []},
        output_dir=str(tmp_path),
        report=lambda *args: None,
    )

    assert concat_calls == [["seedance_shot_01.mp4", "seedance_shot_02_local_scene.mp4"]]
    assert result["succeeded_count"] == 1
    assert result["records"][0]["status"] == "succeeded"


def test_content_repair_reports_unsupported_strategy(monkeypatch, tmp_path):
    calls = []

    monkeypatch.setattr(
        workflow,
        "repair_and_rerender_shot",
        lambda **kwargs: calls.append(kwargs) or {"success": False, "error": "unsupported"},
    )

    result = _repair_rendered_content(
        task_id="task-1",
        repair_records=[{"shot_index": 1, "repair_strategy": "rewrite_shot_goal"}],
        creation_plan={"shots": [{"shot_index": 1}]},
        render_result={"shot_results": []},
        output_dir=str(tmp_path),
        report=lambda *args: None,
    )

    assert calls
    assert result["succeeded_count"] == 0
    assert result["failed_count"] == 1
    assert result["records"][0]["status"] == "failed"


def test_simplify_action_repair_executes_single_shot_rerender(monkeypatch, tmp_path):
    os.environ["ARK_API_KEY"] = "test-key"
    os.environ["ARK_VIDEO_ENDPOINT_ID"] = "ep-video"
    captured_shots = []
    source_clip = tmp_path / "source.mp4"

    def fake_create(api_key, model_endpoint, shot, first_frame_url_override=""):
        captured_shots.append(dict(shot))
        return {"success": True, "task_id": "seedance-task", "error": None}

    monkeypatch.setattr("agent.seedance_video_renderer._create_seedance_task", fake_create)
    monkeypatch.setattr(
        "agent.seedance_video_renderer._poll_seedance_task",
        lambda api_key, seedance_task_id: {
            "success": True,
            "video_url": "https://example.com/video.mp4",
            "error": None,
        },
    )
    monkeypatch.setattr(
        "agent.seedance_video_renderer._download_video",
        lambda video_url, output_path: output_path.write_bytes(b"video") or {"success": True},
    )
    monkeypatch.setattr(
        "agent.seedance_video_renderer._adapt_clip_to_target_duration",
        lambda source_path, output_dir, shot_index, shot: source_path,
    )

    result = repair_and_rerender_shot(
        shot={
            "shot_index": 2,
            "duration_seconds": 3,
            "render_strategy": "image_to_video",
            "product_presence": "required",
            "action": "pour liquid and rotate product",
            "visual_description": "show cup with readable label",
        },
        shot_index=2,
        repair_strategy="simplify_action",
        task_id="task-1",
        output_dir=str(tmp_path),
    )

    assert result["success"] is True
    assert captured_shots
    assert "pour liquid" not in captured_shots[0]["action"]
    assert "No generated text" in captured_shots[0]["video_prompt"]


def test_rewrite_shot_goal_repair_executes_single_shot_rerender(monkeypatch, tmp_path):
    os.environ["ARK_API_KEY"] = "test-key"
    os.environ["ARK_VIDEO_ENDPOINT_ID"] = "ep-video"
    captured_shots = []

    monkeypatch.setattr(
        "agent.seedance_video_renderer._create_seedance_task",
        lambda api_key, model_endpoint, shot, first_frame_url_override="": captured_shots.append(dict(shot))
        or {"success": True, "task_id": "seedance-task", "error": None},
    )
    monkeypatch.setattr(
        "agent.seedance_video_renderer._poll_seedance_task",
        lambda api_key, seedance_task_id: {
            "success": True,
            "video_url": "https://example.com/video.mp4",
            "error": None,
        },
    )
    monkeypatch.setattr(
        "agent.seedance_video_renderer._download_video",
        lambda video_url, output_path: output_path.write_bytes(b"video") or {"success": True},
    )
    monkeypatch.setattr(
        "agent.seedance_video_renderer._adapt_clip_to_target_duration",
        lambda source_path, output_dir, shot_index, shot: source_path,
    )

    result = repair_and_rerender_shot(
        shot={
            "shot_index": 1,
            "duration_seconds": 3,
            "render_strategy": "text_to_video",
            "product_presence": "forbidden",
            "scene_goal": "summer outdoor picnic setup",
            "visual_description": "wrong indoor desk scene",
            "action": "place a glass cup",
        },
        shot_index=1,
        repair_strategy="rewrite_shot_goal",
        task_id="task-1",
        output_dir=str(tmp_path),
    )

    assert result["success"] is True
    assert captured_shots
    assert "summer outdoor picnic setup" in captured_shots[0]["video_prompt"]
    assert "wrong indoor desk scene" not in captured_shots[0]["video_prompt"]


def test_seedance_prompt_removes_generated_cta_text_and_ui_requests():
    prompt = _build_seedance_prompt(
        {
            "render_strategy": "image_to_video",
            "product_presence": "required",
            "visual_description": "用户拿起商品，随后画面定格，叠加购物车图标和“点击下方购买”文字。",
            "action": "放下商品，然后画面出现购物车图标和点击文字。",
            "asset": {"file_path": "/tmp/product.jpg"},
        }
    )

    assert "购物车图标" not in prompt
    assert "点击下方购买" not in prompt
    assert "点击文字" not in prompt
    assert "画面内文字、字符、字幕、水印、UI 元素或额外标签" in prompt


def test_seedance_prompt_does_not_forbid_existing_product_marks_when_visible_marks_exist():
    prompt = _build_seedance_prompt(
        {
            "render_strategy": "image_to_video",
            "product_presence": "required",
            "visual_description": "用户轻触商品，商品自带 CHAKO LAB 字样保持原样。",
            "action": "保持商品稳定。",
            "asset": {"file_path": "/tmp/product.jpg"},
            "product_identity_card": {
                "appearance_summary": "透明水杯，正面有黄色字样",
                "visible_marks": ["CHAKO LAB 字样"],
            },
        }
    )

    assert "新增的非商品自带文字" in prompt
    assert "不要改写商品自带 logo、标识或字样" in prompt
    assert "画面内文字、字符、字幕、水印、UI 元素或额外标签" not in prompt


def test_text_to_video_scene_prompt_removes_specific_brand_text_request():
    prompt = _build_seedance_prompt(
        {
            "render_strategy": "text_to_video",
            "product_presence": "optional",
            "visual_description": "透明杯身与黄色 CHAKO LAB 字样清晰可见，桌面保持干净。",
            "action": "人物拿起水杯。",
        }
    )

    assert "CHAKO LAB" not in prompt
    assert "桌面保持干净" in prompt


def test_create_task_command_accepts_requirement_alignment_fields():
    command = CreateVideoTaskCommand(
        title="laptop video",
        selling_points=["light", "fast"],
        target_platform="tiktok",
        duration_seconds=15,
        style="product_showcase",
        forbidden_changes=["logo", "color"],
        chat_history=["focus on students"],
    )

    task = create_video_task(command, InMemoryTaskRepository())
    task_data = task.to_dict()

    assert task_data["forbidden_changes"] == ["logo", "color"]
    assert task_data["chat_history"] == ["focus on students"]


def test_form_input_splits_comma_and_newline_fields_for_model_consumption():
    assert demo_app._split_selling_points("防漏设计随手包里放, 大容量\n冷热都能装，通勤场景") == [
        "防漏设计随手包里放",
        "大容量",
        "冷热都能装",
        "通勤场景",
    ]


def test_create_page_exposes_custom_inputs_for_open_ended_requirement_fields():
    html = demo_app._render_page()

    assert 'id="product-type-custom"' in html
    assert 'data-custom-target="f-product-type"' in html
    assert 'id="scene-custom"' in html
    assert 'data-custom-target="f-scene"' in html
    assert 'id="audience-custom"' in html
    assert 'data-custom-target="f-audience"' in html
    assert "自定义商品类型" in html
    assert "自定义使用场景" in html
    assert "自定义目标人群" in html


def test_build_product_context_reads_structured_requirements_from_task_data():
    os.environ["AIGC_DISABLE_LLM"] = "1"

    context = build_product_context(
        task_data={
            "title": "laptop",
            "selling_points": ["light"],
            "duration_seconds": 15,
            "structured_requirements": {"target_audience": "students"},
        },
        asset_analysis={"semantic_summary": "", "asset_profiles": [], "product_identity_card": {}},
    )

    assert context["structured_requirements"]["target_audience"] == "students"


def test_build_product_context_promotes_frontend_requirement_fields_to_top_level():
    os.environ["AIGC_DISABLE_LLM"] = "1"

    context = build_product_context(
        task_data={
            "title": "折叠露营灯",
            "product_type": "可折叠磁吸露营灯",
            "selling_points": ["磁吸固定", "帐篷里不占手"],
            "duration_seconds": 15,
            "target_audience": "周末露营新手、车主",
            "usage_scene": "夜间帐篷内照明、车尾收纳",
            "structured_requirements": {
                "target_audience": "周末露营新手、车主",
                "usage_scene": "夜间帐篷内照明、车尾收纳",
                "selling_point_priority": ["磁吸固定", "帐篷里不占手"],
            },
        },
        asset_analysis={"semantic_summary": "", "asset_profiles": [], "product_identity_card": {}},
    )

    assert context["product_type"] == "可折叠磁吸露营灯"
    assert context["target_audience"] == "周末露营新手、车主"
    assert context["usage_scene"] == "夜间帐篷内照明、车尾收纳"
    assert context["audience"] == "周末露营新手、车主"
    assert context["structured_requirements"]["selling_point_priority"] == ["磁吸固定", "帐篷里不占手"]


def test_identity_card_fallback_extracts_constraints_from_product_input():
    os.environ["AIGC_DISABLE_LLM"] = "1"

    identity_card = build_product_identity_card(
        task_data={
            "title": "白色无线鼠标",
            "selling_points": ["轻便", "好看"],
        },
        asset_analysis={
            "assets": [
                {
                    "asset_id": "asset_001",
                    "filename": "mouse.jpg",
                    "asset_type": "image",
                    "is_supported": True,
                    "suggested_role": "商品图或细节图候选",
                }
            ],
            "asset_profiles": [
                {
                    "asset_id": "asset_001",
                    "visual_role": "appearance_anchor",
                    "quality_score": 80,
                }
            ],
        },
    )

    assert identity_card["product_type"] == "鼠标"
    assert identity_card["primary_color"] == "白色"
    assert "保持鼠标商品类型" in identity_card["must_preserve"]
    assert "不能把商品变成其他品类" in identity_card["forbidden_changes"]
    assert identity_card["motion_affordance"]["can_fly"] is False
    assert "飞行" in identity_card["motion_affordance"]["forbidden_actions"]


def test_storyboard_normalization_keeps_state_transition_fields():
    storyboard = _normalize_storyboard(
        [
            {
                "shot_index": 1,
                "duration_seconds": 4,
                "narrative_role": "feature_demo",
                "scene_goal": "证明鼠标轻便好拿",
                "initial_state": "鼠标放在桌面上",
                "action": "一只手拿起鼠标并转向镜头",
                "final_state": "鼠标稳定停在手心",
                "camera_motion": "缓慢推近",
                "product_identity_constraints": ["保持白色鼠标外观"],
                "asset_usage": {
                    "usage_type": "identity_anchor",
                    "required_asset_role": "appearance_anchor",
                    "selected_asset_ids": ["asset_001"],
                    "is_identity_critical": True,
                    "can_generate_without_asset": False,
                    "reason": "展示主体外观",
                },
                "generation_mode": "image_to_video",
                "video_prompt": "展示轻便鼠标",
                "subtitle": "轻便，好拿",
            }
        ]
    )

    shot = storyboard[0]
    assert shot["initial_state"] == "鼠标放在桌面上"
    assert shot["action"] == "一只手拿起鼠标并转向镜头"
    assert shot["final_state"] == "鼠标稳定停在手心"
    assert shot["asset_usage"]["usage_type"] == "identity_anchor"


def test_normalize_script_plan_preserves_rich_story_fields():
    script = _normalize_script_plan(
        {
            "narrative_arc": "hook -> feature -> cta",
            "story_title": "轻薄游戏本",
            "rich_story_text": "先用用户对厚重游戏本的刻板印象切入，再展示真实素材里的黑色机身和绿色标识。",
            "core_message": "轻薄和性能可以同时存在",
            "user_emotion": "打破厚重印象",
            "key_visual_moments": ["A面logo开场", "侧面展示轻薄", "整机定格CTA"],
            "full_subtitle_script": "游戏本也能轻薄。",
            "beats": [
                {
                    "start_seconds": 0,
                    "end_seconds": 3,
                    "role": "hook",
                    "message": "建立反差",
                    "subtitle": "游戏本也能轻薄",
                    "visual_intent": "A面logo开场",
                    "evidence_refs": ["asset:logo"],
                }
            ],
            "hook": "游戏本也能轻薄",
            "body": ["真实机身展示"],
            "cta": "点击了解配置",
        },
        {"duration_seconds": 15},
    )

    assert script["rich_story_text"].startswith("先用用户")
    assert script["core_message"] == "轻薄和性能可以同时存在"
    assert script["key_visual_moments"] == ["A面logo开场", "侧面展示轻薄", "整机定格CTA"]
    assert script["beats"][0]["visual_intent"] == "A面logo开场"
    assert script["beats"][0]["evidence_refs"] == ["asset:logo"]


def test_storyboard_normalization_preserves_asset_id_and_review_fields():
    storyboard = _normalize_storyboard(
        [
            {
                "shot_index": 1,
                "duration_seconds": 3,
                "purpose": "展示logo",
                "narrative_role": "feature_demo",
                "scene_goal": "建立品牌识别",
                "initial_state": "笔记本A面在桌面",
                "action": "镜头推近logo",
                "final_state": "logo停在画面中心",
                "camera_motion": "slow push in",
                "visual_description": "黑色A面和绿色logo",
                "subtitle": "一眼认出",
                "voiceover": "一眼认出",
                "asset_id": "asset_logo",
                "asset_requirement": "A面logo图",
                "product_presence": "required",
                "identity_strictness": "high",
                "forbidden_variation": ["不要改变logo"],
                "review_focus": ["logo一致性"],
                "completion_criteria": ["logo清晰可见"],
            }
        ]
    )

    shot = storyboard[0]
    assert shot["asset_id"] == "asset_logo"
    assert shot["asset_usage"]["selected_asset_ids"] == ["asset_logo"]
    assert shot["forbidden_variation"] == ["不要改变logo"]
    assert shot["review_focus"] == ["logo一致性"]
    assert shot["completion_criteria"] == ["logo清晰可见"]


def test_storyboard_normalization_preserves_continuity_mode():
    storyboard = _normalize_storyboard(
        [
            {
                "shot_index": 1,
                "duration_seconds": 2,
                "narrative_role": "product_reveal",
                "continuity_mode": "shared_scene_bridge",
            }
        ]
    )

    assert storyboard[0]["continuity_mode"] == "shared_scene_bridge"


def test_fallback_director_storyboard_includes_story_scene_reveal_and_cta():
    storyboard = _fallback_director_storyboard(
        product_context={
            "duration_seconds": 15,
            "product_title": "轻薄笔记本",
            "product_identity_card": {"must_preserve": ["保持黑色机身"]},
        },
        script_plan={
            "hook": "通勤包不该被电脑塞满",
            "body": ["轻薄机身", "磨砂质感"],
            "cta": "点击了解更多",
        },
        asset_analysis={
            "assets": [
                {
                    "asset_id": "asset_001",
                    "asset_type": "image",
                    "is_supported": True,
                    "file_path": "/tmp/laptop.jpg",
                }
            ],
            "asset_profiles": [],
        },
    )

    roles = [shot["narrative_role"] for shot in storyboard]
    assert roles == ["hook", "product_reveal", "feature_demo", "detail_proof", "cta"]
    assert storyboard[0]["render_strategy"] == "text_to_video"
    assert storyboard[1]["continuity_mode"] == ""
    assert storyboard[1]["asset_id"] == "asset_001"
    assert storyboard[1]["transition_type"] == "crossfade"
    assert storyboard[2]["asset_id"] == "asset_001"


def test_match_assets_prefers_storyboard_asset_id():
    matches = match_assets_to_storyboard(
        storyboard=[
            {
                "shot_index": 1,
                "asset_id": "asset_second",
                "render_strategy": "image_to_video",
                "product_presence": "required",
                "identity_strictness": "high",
            }
        ],
        asset_analysis={
            "assets": [
                {
                    "asset_id": "asset_first",
                    "asset_type": "image",
                    "is_supported": True,
                    "file_path": "/tmp/first.jpg",
                },
                {
                    "asset_id": "asset_second",
                    "asset_type": "image",
                    "is_supported": True,
                    "file_path": "/tmp/second.jpg",
                },
            ]
        },
    )

    assert matches[0]["matched_asset"]["asset_id"] == "asset_second"
    assert matches[0]["strategy"] == "image_to_video"
    assert matches[0]["render_input"]["file_path"] == "/tmp/second.jpg"


def _valid_storyboard_shot(index: int) -> dict:
    return {
        "shot_index": index,
        "duration_seconds": 2,
        "purpose": f"shot {index}",
        "narrative_role": "feature_demo",
        "scene_goal": f"goal {index}",
        "initial_state": "product on desk",
        "action": "slow push in",
        "final_state": "product remains stable",
        "camera_motion": "slow push in",
        "visual_description": "show product clearly",
        "subtitle": f"shot {index}",
        "voiceover": f"shot {index}",
        "asset_requirement": "product image",
        "render_strategy": "image_to_video",
        "product_presence": "required",
        "identity_strictness": "high",
        "forbidden_variation": ["do not change logo"],
        "review_focus": ["logo consistency"],
    }


def test_storyboard_review_rejects_too_few_shots():
    review = review_storyboard(
        [_valid_storyboard_shot(1), _valid_storyboard_shot(2)],
        {"duration_seconds": 15},
    )

    assert review["passed"] is False
    assert any("3-7" in issue for issue in review["issues"])


def test_storyboard_review_rejects_too_many_shots():
    review = review_storyboard(
        [_valid_storyboard_shot(index) for index in range(1, 9)],
        {"duration_seconds": 15},
    )

    assert review["passed"] is False
    assert any("3-7" in issue for issue in review["issues"])


def test_storyboard_review_rejects_all_product_motion_shots_without_story_scene():
    storyboard = [_valid_storyboard_shot(index) for index in range(1, 5)]
    storyboard[-1]["narrative_role"] = "cta"

    review = review_storyboard(storyboard, {"duration_seconds": 15})

    assert review["passed"] is False
    assert any("剧情场景" in issue for issue in review["issues"])


def test_seedance_prompt_includes_identity_and_motion_constraints():
    identity_card = {
        "appearance_summary": "白色无线鼠标，扁平椭圆轮廓，带滚轮和左右按键",
        "must_preserve": ["白色主体", "滚轮位置", "鼠标形态"],
        "forbidden_changes": ["不能变成键盘", "不能改变主色", "不能增加屏幕"],
        "motion_affordance": {
            "allowed_actions": ["手持展示", "桌面滑动"],
            "forbidden_actions": ["飞行", "展开变形"],
        },
    }
    creation_plan = build_creation_plan(
        product_context={
            "target_platform": "tiktok",
            "product_identity_card": identity_card,
        },
        storyboard=[
            {
                "shot_index": 1,
                "duration_seconds": 4,
                "purpose": "证明鼠标轻便好拿",
                "visual_description": "白色鼠标在桌面上",
                "initial_state": "鼠标放在桌面上",
                "action": "一只手拿起鼠标",
                "final_state": "鼠标停在手心",
                "subtitle": "轻便，好拿",
                "voiceover": "轻便，好拿",
                "product_presence": "required",
                "identity_strictness": "high",
            }
        ],
        asset_matching=[
            {
                "shot_index": 1,
                "strategy": "text_to_video",
                "matched_asset": None,
                "note": "无素材，文生视频",
            }
        ],
    )

    prompt = _build_seedance_prompt(creation_plan["shots"][0])

    assert "生成方式：文生视频" in prompt
    assert "镜头目标：" in prompt
    assert "白色无线鼠标" in prompt
    assert "不能变成键盘" in prompt
    assert "一只手拿起鼠标" in prompt
    assert "严禁出现烟雾、雾气、蒸汽、尘埃、光束或粒子飘浮" in prompt
    assert len(prompt) <= 900
    assert "--ratio" not in prompt
    assert "--dur" not in prompt
    assert "--rs" not in prompt


def test_seedance_prompt_uses_static_camera_for_brand_detail_image_to_video():
    prompt = _build_seedance_prompt(
        {
            "shot_index": 1,
            "purpose": "展示雷蛇笔记本，商标不能变，Logo 位置不能变",
            "visual_description": "Razer laptop with visible logo",
            "action": "slow push-in",
            "camera_motion": "slow push-in",
            "render_strategy": "image_to_video",
            "asset": {"file_path": "/tmp/reference.jpg"},
            "product_identity_card": {
                "appearance_summary": "雷蛇笔记本",
                "visible_marks": ["绿色三头蛇 logo"],
                "must_preserve": ["商标不能变", "Logo 位置"],
                "forbidden_changes": ["不能增加 logo文字"],
            },
        }
    )

    assert "生成方式：图生视频" in prompt
    assert "商品身份约束：" in prompt
    assert "雷蛇" in prompt
    assert "Razer" not in prompt
    assert "绿色三头蛇" not in prompt
    assert "保持首帧已有商品标识稳定，不新增、改写或重新绘制文字" in prompt
    assert "logo文字" not in prompt
    assert "上传素材作为首帧" in prompt
    assert "镜头动作：定镜，保持构图稳定，不推近、不拉远、不旋转、不环绕" in prompt
    assert len(prompt) <= 900


def test_seedance_renderer_reads_asset_from_render_input_contract():
    asset = _resolve_seedance_asset(
        {
            "render_strategy": "image_to_video",
            "render_input": {
                "type": "asset",
                "asset_id": "asset_logo",
                "file_path": "/tmp/logo.jpg",
            },
        }
    )

    assert asset["asset_id"] == "asset_logo"
    assert asset["file_path"] == "/tmp/logo.jpg"


def test_text_to_video_scene_prompt_does_not_request_brand_details():
    prompt = _build_seedance_prompt(
        {
            "shot_index": 1,
            "purpose": "建立宿舍桌面场景",
            "visual_description": "干净书桌、书本和台灯，不展示具体商品",
            "action": "灯光保持稳定",
            "camera_motion": "轻微水平平移",
            "render_strategy": "text_to_video",
            "product_presence": "optional",
            "identity_strictness": "low",
            "product_identity_card": {
                "appearance_summary": "黑色雷蛇笔记本",
                "visible_marks": ["绿色三头蛇 logo"],
                "must_preserve": ["保持绿色三头蛇 logo"],
            },
        }
    )

    assert "绿色三头蛇" not in prompt
    assert "雷蛇" not in prompt
    assert "不展示可识别商品主体" in prompt
    assert "不出现 logo、品牌标识或可读文字" in prompt


def test_image_to_video_scene_background_prompt_does_not_inject_product_identity():
    prompt = _build_seedance_prompt(
        {
            "shot_index": 1,
            "purpose": "在统一棚拍背景上建立开场",
            "visual_description": "保持简洁墙面和桌面，不展示商品",
            "action": "光线轻微变化",
            "render_strategy": "image_to_video",
            "product_presence": "forbidden",
            "asset": {
                "file_path": "/tmp/studio_background.jpg",
                "is_scene_background": True,
            },
            "product_identity_card": {
                "appearance_summary": "黑色雷蛇笔记本",
                "visible_marks": ["绿色三头蛇 logo"],
            },
        }
    )

    assert "保持首帧的墙面、桌面、光线和构图" in prompt
    assert "不新增商品主体" in prompt
    assert "雷蛇" not in prompt
    assert "绿色三头蛇" not in prompt


def test_scene_background_shot_uses_local_renderer():
    shot = {
        "render_strategy": "image_to_video",
        "render_input": {
            "type": "asset",
            "file_path": "/tmp/studio_background.jpg",
            "is_scene_background": True,
        },
    }

    assert _should_render_scene_background_locally(shot) is True


def test_local_scene_background_renderer_keeps_scene_without_video_model(tmp_path):
    source_path = tmp_path / "studio_background.jpg"
    reveal_path = tmp_path / "studio_anchor.jpg"
    output_path = tmp_path / "scene_clip.mp4"
    Image.new("RGB", (72, 128), (236, 234, 229)).save(source_path)
    Image.new("RGB", (72, 128), (220, 20, 20)).save(reveal_path)

    result = _render_local_scene_background_clip(
        {
            "duration_seconds": 1,
            "render_input": {
                "type": "asset",
                "file_path": str(source_path),
                "is_scene_background": True,
                "reveal_asset_path": str(reveal_path),
            },
        },
        output_path,
    )

    assert result["success"] is True
    assert output_path.exists()
    import imageio.v2 as imageio

    reader = imageio.get_reader(output_path)
    first_frame = np.asarray(reader.get_data(0))
    last_frame = np.asarray(reader.get_data(23))
    reader.close()
    assert int(last_frame[:, :, 1].mean()) < int(first_frame[:, :, 1].mean())
    assert int(last_frame[:, :, 0].mean()) > int(last_frame[:, :, 1].mean()) + 150


def test_studio_anchor_composes_transparent_product_on_consistent_background():
    from PIL import Image

    foreground = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    foreground.putpixel((10, 10), (255, 0, 0, 255))

    anchor = _compose_studio_anchor(foreground, target_size=(100, 160))

    assert anchor.mode == "RGB"
    assert anchor.size == (100, 160)
    # 统一背景不是透明图，也不是原始照片背景。
    assert anchor.getpixel((0, 0)) == (236, 234, 229)
    # 前景商品仍然存在于合成结果中。
    assert any(pixel[0] > 240 and pixel[1] < 80 for pixel in anchor.get_flattened_data())


def test_background_removal_rejects_nearly_transparent_foreground():
    from PIL import Image

    weak_foreground = Image.new("RGBA", (40, 40), (20, 20, 20, 10))
    solid_foreground = Image.new("RGBA", (40, 40), (20, 20, 20, 255))

    assert _foreground_is_usable(weak_foreground) is False
    assert _foreground_is_usable(solid_foreground) is True


def test_frame_crossfade_blends_adjacent_clips_without_black_frame():
    previous_tail = [
        np.full((1, 1, 3), 20, dtype=np.uint8),
        np.full((1, 1, 3), 40, dtype=np.uint8),
    ]
    next_head = [
        np.full((1, 1, 3), 220, dtype=np.uint8),
        np.full((1, 1, 3), 240, dtype=np.uint8),
    ]

    blended = _blend_transition_frames(previous_tail, next_head)

    assert len(blended) == 2
    assert 20 < int(blended[0][0, 0, 0]) < 220
    assert 40 < int(blended[1][0, 0, 0]) < 240


def test_render_segment_adapter_keeps_llm_duration_and_model_duration_separate():
    storyboard = [
        {"shot_index": 1, "duration_seconds": 3, "subtitle": "shot one"},
        {"shot_index": 2, "duration_seconds": 5, "subtitle": "shot two"},
        {"shot_index": 3, "duration_seconds": 4, "subtitle": "shot three"},
    ]

    segments = adapt_storyboard_to_render_segments(storyboard)

    assert [segment["target_duration_seconds"] for segment in segments] == [3.0, 5.0, 4.0]
    assert all(segment["model_duration_seconds"] == 5 for segment in segments)
    assert segments[0]["timeline_start_seconds"] == 0.0
    assert segments[2]["timeline_end_seconds"] == 12.0


def test_subtitle_timeline_uses_trimmed_render_segment_duration():
    shots = [
        {
            "shot_index": 1,
            "duration_seconds": 3,
            "subtitle": "shot one",
            "render_segment": {"target_duration_seconds": 3},
        },
        {
            "shot_index": 2,
            "duration_seconds": 4,
            "subtitle": "shot two",
            "render_segment": {"target_duration_seconds": 4},
        },
    ]

    timeline = _build_subtitle_timeline(shots)

    assert timeline[0]["start"] == 0.0
    assert timeline[0]["end"] == 3.0
    assert timeline[1]["start"] == 3.0
    assert timeline[1]["end"] == 7.0

def test_asset_analysis_for_llm_hides_filename_and_paths():
    safe_analysis = _asset_analysis_for_llm(
        {
            "asset_count": 1,
            "supported_count": 1,
            "assets": [
                {
                    "asset_id": "asset_001",
                    "filename": "mouse_product_white.jpg",
                    "file_path": "/tmp/mouse_product_white.jpg",
                    "public_url": "/uploads/mouse_product_white.jpg",
                    "asset_type": "image",
                    "content_type": "image/jpeg",
                    "file_size": 10,
                    "is_supported": True,
                }
            ],
            "asset_profiles": [
                {
                    "asset_id": "asset_001",
                    "filename": "mouse_product_white.jpg",
                    "visual_role": "appearance_anchor",
                }
            ],
        }
    )

    serialized = str(safe_analysis)
    assert "mouse_product_white.jpg" not in serialized
    assert "/tmp/" not in serialized
    assert "/uploads/" not in serialized
    assert safe_analysis["assets"][0]["asset_id"] == "asset_001"


def test_identity_card_fallback_does_not_infer_brand_from_filename():
    os.environ["AIGC_DISABLE_LLM"] = "1"

    identity_card = build_product_identity_card(
        task_data={
            "title": "laptop",
            "selling_points": ["lightweight"],
        },
        asset_analysis={
            "assets": [
                {
                    "asset_id": "asset_001",
                    "filename": "razer_brand_laptop.jpg",
                    "asset_type": "image",
                    "file_path": "",
                    "is_supported": True,
                }
            ],
            "asset_profiles": [
                {
                    "asset_id": "asset_001",
                    "filename": "razer_brand_laptop.jpg",
                    "visual_role": "appearance_anchor",
                }
            ],
        },
    )

    serialized = str(identity_card).lower()
    assert "razer" not in serialized
    assert "brand" not in serialized
    assert identity_card["identity_confidence"] == "low"


def test_identity_critical_shot_prefers_real_asset_even_if_llm_suggests_text_to_video():
    assert _shot_prefers_real_asset(
        {
            "render_strategy": "text_to_video",
            "asset_usage": {
                "usage_type": "identity_anchor",
                "is_identity_critical": True,
            },
        }
    )


def test_multimodal_call_retries_with_compressed_jpeg(monkeypatch, tmp_path):
    from PIL import Image

    image_path = tmp_path / "product.png"
    Image.new("RGBA", (1200, 800), (20, 80, 160, 255)).save(image_path)
    calls = []

    monkeypatch.delenv("AIGC_DISABLE_LLM", raising=False)
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    monkeypatch.setenv("ARK_TEXT_ENDPOINT_ID", "test-endpoint")

    def fake_post(base_url, api_key, payload, purpose, timeout=30):
        calls.append(payload)
        if len(calls) == 1:
            return {"ok": False, "content": "", "error": "payload too large"}
        return {"ok": True, "content": "ok", "error": None}

    monkeypatch.setattr(workflow, "_post_openai_compatible_payload", fake_post)

    result = workflow._call_multimodal_llm(
        prompt_data={"task": "describe image"},
        image_paths=[str(image_path)],
        purpose="unit_test_multimodal_retry",
    )

    assert result["ok"] is True
    assert len(calls) == 2
    first_url = calls[0]["messages"][1]["content"][1]["image_url"]["url"]
    second_url = calls[1]["messages"][1]["content"][1]["image_url"]["url"]
    assert first_url.startswith("data:image/png;base64,")
    assert second_url.startswith("data:image/jpeg;base64,")


def test_seedance_fallback_marks_final_check_as_needs_review():
    final_check = run_final_check(
        product_context={"duration_seconds": 5},
        storyboard=[{"shot_index": 1, "visual_description": "show product", "subtitle": "light"}],
        creation_plan={"total_duration_seconds": 5},
        render_result={
            "success": True,
            "fallback_from": {"error": "Seedance Invalid content.text"},
        },
    )

    assert final_check["passed"] is False
    assert any("Seedance" in issue and "fallback" in issue for issue in final_check["issues"])


def test_local_fallback_preview_text_contains_storyboard_fields():
    preview_text = _build_storyboard_preview_text(
        {
            "shot_index": 2,
            "scene_goal": "展示轻便卖点",
            "initial_state": "笔记本放在桌面",
            "action": "单手拿起",
            "final_state": "突出轻薄机身",
            "visual_description": "侧面低角度特写",
        }
    )

    assert "2" in preview_text
    assert "展示轻便卖点" in preview_text
    assert "单手拿起" in preview_text
    assert "突出轻薄机身" in preview_text


def test_content_review_failure_blocks_final_check(monkeypatch, tmp_path):
    reference_image = tmp_path / "reference.jpg"
    generated_frame = tmp_path / "frame.jpg"
    reference_image.write_bytes(b"reference")
    generated_frame.write_bytes(b"frame")

    creation_plan = {
        "total_duration_seconds": 4,
        "shots": [
            {
                "shot_index": 1,
                "duration_seconds": 4,
                "visual_description": "show product",
                "subtitle": "light",
                "asset": {"asset_type": "image", "file_path": str(reference_image)},
            }
        ],
    }

    monkeypatch.setattr(
        workflow,
        "_extract_video_review_frames",
        lambda **kwargs: [
            {"shot_index": 1, "frame_path": str(generated_frame), "timestamp_seconds": 2.0}
        ],
    )
    monkeypatch.setattr(
        workflow,
        "_call_multimodal_llm",
        lambda prompt_data, image_paths, purpose: {
            "ok": True,
            "content": '{"passed": false, "shot_reviews": [{"shot_index": 1, "pass": false, "failed_dimensions": ["product_consistency"], "main_issue": "different product", "repair_strategy": "rerender_with_identity_anchor"}]}',
            "error": None,
        },
    )

    content_review = review_rendered_video_content(
        product_context={"product_identity_card": {"appearance_summary": "reference laptop"}},
        creation_plan=creation_plan,
        render_result={"success": True, "video_path": str(tmp_path / "video.mp4")},
        output_dir=str(tmp_path),
    )
    final_check = run_final_check(
        product_context={"duration_seconds": 4},
        storyboard=[{"shot_index": 1, "visual_description": "show product", "subtitle": "light"}],
        creation_plan=creation_plan,
        render_result={"success": True},
        content_review=content_review,
    )

    assert content_review["passed"] is False
    assert content_review["repair_records"][0]["shot_index"] == 1
    assert final_check["passed"] is False
    assert any("内容审视" in issue for issue in final_check["issues"])


def test_content_review_uses_uploaded_reference_images_when_shots_do_not_bind_asset(monkeypatch, tmp_path):
    reference_image = tmp_path / "uploaded_reference.jpg"
    generated_frame = tmp_path / "frame.jpg"
    reference_image.write_bytes(b"reference")
    generated_frame.write_bytes(b"frame")

    creation_plan = {
        "total_duration_seconds": 4,
        "shots": [
            {
                "shot_index": 1,
                "duration_seconds": 4,
                "visual_description": "show product with text prompt only",
                "subtitle": "light",
            }
        ],
    }

    monkeypatch.setattr(
        workflow,
        "_extract_video_review_frames",
        lambda **kwargs: [
            {"shot_index": 1, "frame_path": str(generated_frame), "timestamp_seconds": 2.0}
        ],
    )

    captured_image_paths = []

    def fake_multimodal_llm(prompt_data, image_paths, purpose):
        captured_image_paths.extend(image_paths)
        return {
            "ok": True,
            "content": '{"passed": true, "shot_reviews": [{"shot_index": 1, "pass": true, "failed_dimensions": [], "main_issue": "", "repair_strategy": ""}]}',
            "error": None,
        }

    monkeypatch.setattr(workflow, "_call_multimodal_llm", fake_multimodal_llm)

    content_review = review_rendered_video_content(
        product_context={
            "product_identity_card": {"appearance_summary": "reference laptop"},
            "reference_image_paths": [str(reference_image)],
        },
        creation_plan=creation_plan,
        render_result={"success": True, "video_path": str(tmp_path / "video.mp4")},
        output_dir=str(tmp_path),
    )

    assert content_review["skipped"] is False
    assert str(reference_image) in captured_image_paths


def test_build_creation_plan_accepts_storyboard_without_legacy_purpose():
    storyboard = [
        {
            "shot_index": 1,
            "duration_seconds": 4,
            "narrative_role": "hook",
            "scene_goal": "用轻薄机身吸引注意",
            "initial_state": "笔记本放在桌面",
            "action": "镜头推近机身侧面",
            "final_state": "突出轻薄外观",
            "visual_description": "桌面上的笔记本电脑",
            "subtitle": "轻薄，也能高性能",
            "voiceover": "轻薄，也能高性能",
        }
    ]

    plan = build_creation_plan(
        product_context={"target_platform": "tiktok", "product_identity_card": {}},
        storyboard=storyboard,
        asset_matching=[{"shot_index": 1, "strategy": "text_to_video"}],
    )

    assert plan["shots"][0]["purpose"] == "用轻薄机身吸引注意"
    assert plan["shots"][0]["scene_goal"] == "用轻薄机身吸引注意"


def test_build_creation_plan_resolves_asset_id_to_render_input_file_path():
    storyboard = [
        {
            "shot_index": 1,
            "duration_seconds": 4,
            "purpose": "show real laptop logo",
            "narrative_role": "detail_proof",
            "scene_goal": "prove product identity",
            "initial_state": "closed laptop on desk",
            "action": "camera pushes toward the logo",
            "final_state": "logo area remains visible",
            "camera_motion": "slow push in",
            "visual_description": "close shot of laptop logo",
            "subtitle": "real product",
            "voiceover": "real product",
            "asset_id": "asset_logo",
        }
    ]

    plan = build_creation_plan(
        product_context={"target_platform": "tiktok", "product_identity_card": {}},
        storyboard=storyboard,
        asset_matching=[
            {
                "shot_index": 1,
                "strategy": "image_to_video",
                "matched_asset": {
                    "asset_id": "asset_logo",
                    "asset_type": "image",
                    "file_path": "/tmp/logo.jpg",
                },
                "render_input": {
                    "type": "asset",
                    "asset_id": "asset_logo",
                    "asset_type": "image",
                    "file_path": "/tmp/logo.jpg",
                    "render_mode": "image_to_video",
                    "fallback_policy": "local_asset_motion",
                },
            }
        ],
    )

    shot = plan["shots"][0]
    assert shot["asset_id"] == "asset_logo"
    assert shot["asset"]["file_path"] == "/tmp/logo.jpg"
    assert shot["render_input"]["type"] == "asset"
    assert shot["render_input"]["file_path"] == "/tmp/logo.jpg"
    assert shot["render_strategy"] == "image_to_video"


def test_build_creation_plan_forbids_text_to_video_when_matched_asset_exists():
    storyboard = [
        {
            "shot_index": 1,
            "duration_seconds": 4,
            "purpose": "show real laptop shell",
            "narrative_role": "feature_demo",
            "scene_goal": "keep product appearance stable",
            "initial_state": "laptop is closed",
            "action": "camera moves across the shell",
            "final_state": "shell shape stays unchanged",
            "camera_motion": "slow lateral move",
            "visual_description": "real laptop shell",
            "subtitle": "stable look",
            "voiceover": "stable look",
            "asset_id": "asset_shell",
        }
    ]

    plan = build_creation_plan(
        product_context={"target_platform": "tiktok", "product_identity_card": {}},
        storyboard=storyboard,
        asset_matching=[
            {
                "shot_index": 1,
                # 上游模型有时会误判成纯文本生成；只要已经匹配到真实素材，执行层就必须改成图生视频。
                "strategy": "text_to_video",
                "matched_asset": {
                    "asset_id": "asset_shell",
                    "asset_type": "image",
                    "file_path": "/tmp/shell.jpg",
                },
            }
        ],
    )

    shot = plan["shots"][0]
    assert shot["render_strategy"] == "image_to_video"
    assert shot["render_input"]["type"] == "asset"
    assert shot["render_input"]["asset_id"] == "asset_shell"
    assert shot["render_input"]["file_path"] == "/tmp/shell.jpg"


def test_build_creation_plan_rebuilds_stale_text_render_input_when_asset_matched():
    storyboard = [
        {
            "shot_index": 1,
            "duration_seconds": 4,
            "purpose": "show real laptop logo",
            "narrative_role": "detail_proof",
            "scene_goal": "keep product identity",
            "initial_state": "closed laptop on desk",
            "action": "hold the composition",
            "final_state": "logo remains stable",
            "camera_motion": "static",
            "visual_description": "real laptop logo",
            "subtitle": "real identity",
            "voiceover": "real identity",
            "asset_id": "asset_logo",
        }
    ]

    plan = build_creation_plan(
        product_context={"target_platform": "tiktok", "product_identity_card": {}},
        storyboard=storyboard,
        asset_matching=[
            {
                "shot_index": 1,
                "strategy": "text_to_video",
                "matched_asset": {
                    "asset_id": "asset_logo",
                    "asset_type": "image",
                    "file_path": "/tmp/logo.jpg",
                },
                "render_input": {
                    "type": "text",
                    "prompt": "stale text prompt",
                },
            }
        ],
    )

    shot = plan["shots"][0]
    assert shot["render_strategy"] == "image_to_video"
    assert shot["render_input"]["type"] == "asset"
    assert shot["render_input"]["asset_id"] == "asset_logo"
    assert shot["render_input"]["file_path"] == "/tmp/logo.jpg"


def test_build_creation_plan_preserves_forced_video_prompt_contract():
    storyboard = [
        {
            "shot_index": 1,
            "duration_seconds": 5,
            "purpose": "test ideal commerce prompt",
            "narrative_role": "commerce_result_scene",
            "scene_goal": "用新场景结果证明卖点",
            "initial_state": "硬切到办公楼入口的新场景。",
            "action": "水杯已经在背包侧袋中，人物整理肩带。",
            "final_state": "商品结果状态清楚。",
            "camera_motion": "定镜",
            "visual_description": "这段不应被二次改写。",
            "subtitle": "通勤随手带",
            "voiceover": "通勤随手带",
            "render_strategy": "text_to_video",
            "product_presence": "optional",
            "identity_strictness": "medium",
            "force_video_prompt": True,
            "video_prompt": "这是 5 秒文生视频，是硬切后的新镜头，不承接上一镜的时间、地点或背景。",
            "planner_source": "B_ideal_commerce_scene:single_scene_prompt_montage",
            "material_strategy": "ideal_commerce_scene",
        }
    ]

    plan = build_creation_plan(
        product_context={"target_platform": "tiktok", "product_identity_card": {}},
        storyboard=storyboard,
        asset_matching=[{"shot_index": 1, "strategy": "text_to_video"}],
    )

    shot = plan["shots"][0]
    assert shot["render_strategy"] == "text_to_video"
    assert shot["force_video_prompt"] is True
    assert shot["video_prompt"].startswith("这是 5 秒文生视频")


def test_identity_and_context_tolerate_list_shaped_model_output():
    task_data = {
        "title": "笔记本电脑",
        "selling_points": ["轻薄"],
        "structured_requirements": ["商标不能变"],
        "duration_seconds": 15,
    }
    asset_analysis = {
        "assets": [],
        "asset_profiles": [],
        "product_identity_card": ["模型错误地返回了数组"],
    }

    identity_card = build_product_identity_card(task_data, asset_analysis, structured_requirements=["商标不能变"])
    asset_analysis["product_identity_card"] = asset_analysis["product_identity_card"]
    product_context = build_product_context(task_data, asset_analysis)

    assert isinstance(identity_card, dict)
    assert isinstance(product_context["product_identity_card"], dict)
    assert product_context["motion_affordance"] == {}


def test_save_workflow_artifacts_ignores_list_llm_sources(tmp_path):
    artifacts_dir = _save_workflow_artifacts(
        task_id="task_test",
        output_dir=str(tmp_path),
        artifacts={
            "workflow_status": "needs_review",
            "workflow_stage": "draft_needs_review",
            "storyboard": [{"shot_index": 1, "subtitle": "test"}],
            "asset_matching": [{"shot_index": 1, "strategy": "image_to_video"}],
            "render_result": {"shot_results": [{"shot_index": 1}]},
        },
    )

    assert list(Path(artifacts_dir).glob("*_08_storyboard.json"))
    assert list(Path(artifacts_dir).glob("*_workflow_trace.json"))


def test_normalize_storyboard_preserves_continuity_fields():
    storyboard = _normalize_storyboard(
        [
            {
                "shot_index": 1,
                "duration_seconds": 3,
                "scene_goal": "keep the same desk scene",
                "continuity_group": "desk_story",
                "transition_type": "continue_from_previous",
            }
        ]
    )

    assert storyboard[0]["continuity_group"] == "desk_story"
    assert storyboard[0]["transition_type"] == "continue_from_previous"


def test_build_creation_plan_injects_shared_visual_style_bible():
    visual_style_bible = {
        "realism": "photorealistic commercial video",
        "lighting": "soft daylight from camera left",
    }
    plan = build_creation_plan(
        product_context={
            "target_platform": "tiktok",
            "product_identity_card": {},
            "visual_style_bible": visual_style_bible,
        },
        storyboard=[
            {
                "shot_index": 1,
                "duration_seconds": 3,
                "scene_goal": "keep the same desk scene",
                "visual_description": "same desk",
                "subtitle": "same style",
                "voiceover": "same style",
                "continuity_group": "desk_story",
                "transition_type": "hard_cut",
            }
        ],
        asset_matching=[{"shot_index": 1, "strategy": "text_to_video"}],
    )

    assert plan["visual_style_bible"] == visual_style_bible
    assert plan["shots"][0]["visual_style_bible"] == visual_style_bible
    assert plan["shots"][0]["continuity_group"] == "desk_story"
    assert plan["shots"][0]["transition_type"] == "hard_cut"


def test_script_review_rejects_slogan_only_plan_without_rich_story():
    review = review_script_plan(
        {
            "hook": "轻薄高性能",
            "body": ["通勤更方便"],
            "cta": "点击了解更多",
            "tone": "真实写实",
            "rich_story_text": "轻薄笔记本，值得拥有。",
            "beats": [
                {
                    "role": "hook",
                    "message": "轻薄高性能",
                    "subtitle": "轻薄高性能",
                    "visual_intent": "",
                },
                {
                    "role": "cta",
                    "message": "点击了解更多",
                    "subtitle": "点击了解更多",
                    "visual_intent": "",
                },
            ],
        }
    )

    assert review["passed"] is False
    assert any("rich_story_text" in issue for issue in review["issues"])
    assert any("visual_intent" in issue for issue in review["issues"])


def test_ensure_storyboard_continuity_repairs_invalid_previous_tail_reference():
    storyboard = _ensure_storyboard_continuity(
        [
            {
                "shot_index": 1,
                "narrative_role": "problem",
                "transition_type": "continue_from_previous",
                "continuity_group": "desk_story",
                "product_presence": "forbidden",
            },
            {
                "shot_index": 2,
                "narrative_role": "feature_demo",
                "transition_type": "continue_from_previous",
                "continuity_group": "product_showcase",
                "product_presence": "required",
                "asset_id": "asset-product",
            },
            {
                "shot_index": 3,
                "narrative_role": "detail_proof",
                "transition_type": "continue_from_previous",
                "continuity_group": "product_showcase",
                "product_presence": "required",
                "asset_id": "asset-product",
            },
        ]
    )

    assert storyboard[0]["transition_type"] == "hard_cut"
    assert storyboard[1]["transition_type"] == "hard_cut"
    assert storyboard[2]["transition_type"] == "continue_from_previous"
    assert storyboard[2]["anchor_last_frame"] is True


def test_fallback_director_storyboard_contains_safe_continuity_groups():
    storyboard = _fallback_director_storyboard(
        product_context={
            "duration_seconds": 15,
            "product_identity_card": {
                "must_preserve": ["keep shell"],
                "reference_asset_ids": ["asset-product"],
            },
        },
        script_plan={
            "hook": "通勤包已经装不下",
            "body": ["轻薄机身", "磨砂质感"],
            "cta": "点击了解更多",
        },
        asset_analysis={
            "assets": [
                {
                    "asset_id": "asset-product",
                    "asset_type": "image",
                    "file_path": "/tmp/product.jpg",
                    "is_supported": True,
                }
            ],
            "asset_profiles": [],
        },
    )

    product_shots = [
        shot for shot in storyboard
        if shot.get("narrative_role") in {"feature_demo", "detail_proof", "cta"}
    ]
    assert product_shots[0]["transition_type"] == "hard_cut"
    assert all(shot["continuity_group"] == "product_showcase" for shot in product_shots)
    assert product_shots[1]["transition_type"] == "continue_from_previous"
    assert product_shots[1]["anchor_last_frame"] is True


def test_fallback_script_remains_rich_enough_for_director_review(monkeypatch):
    monkeypatch.setattr(
        "agent.video_generation_workflow._call_text_llm",
        lambda *args, **kwargs: {"ok": False, "content": "", "error": "disabled"},
    )

    script = plan_script(
        product_context={
            "duration_seconds": 15,
            "product_title": "轻薄笔记本",
            "selling_points": ["方便通勤", "磨砂质感"],
            "product_identity_card": {"appearance_summary": "黑色磨砂笔记本"},
        },
        director_decision={},
    )

    review = review_script_plan(script)

    assert review["passed"] is True
    assert all(beat["visual_intent"] for beat in script["beats"])


def test_storyboard_review_rejects_invalid_continuity_and_crossfade_abuse():
    storyboard = []
    for index, role in enumerate(["problem", "product_reveal", "feature_demo", "cta"], start=1):
        storyboard.append(
            {
                "shot_index": index,
                "duration_seconds": 3,
                "purpose": role,
                "narrative_role": role,
                "scene_goal": role,
                "initial_state": "start",
                "action": "act",
                "final_state": "end",
                "camera_motion": "static",
                "visual_description": role,
                "subtitle": role,
                "voiceover": role,
                "asset_requirement": "asset" if role in {"feature_demo", "cta"} else "none",
                "render_strategy": "image_to_video" if role in {"feature_demo", "cta"} else "text_to_video",
                "product_presence": "required" if role in {"feature_demo", "cta"} else "forbidden",
                "identity_strictness": "high" if role in {"feature_demo", "cta"} else "low",
                "forbidden_variation": ["keep shell"] if role in {"feature_demo", "cta"} else [],
                "review_focus": ["identity"] if role in {"feature_demo", "cta"} else ["story"],
                "transition_type": "crossfade",
                "continuity_group": "",
            }
        )
    storyboard[2]["transition_type"] = "continue_from_previous"

    review = review_storyboard(storyboard, {"duration_seconds": 15})

    assert review["passed"] is False
    assert any("continue_from_previous" in issue for issue in review["issues"])
    assert any("crossfade" in issue for issue in review["issues"])


def test_director_storyboard_receives_rich_script_and_preserves_continuity(monkeypatch):
    captured_prompt = {}

    def fake_text_llm(prompt, purpose, temperature):
        captured_prompt.update(prompt)
        return {
            "ok": True,
            "content": """
            [
              {
                "shot_index": 1,
                "duration_seconds": 3,
                "purpose": "establish commute problem",
                "narrative_role": "problem",
                "scene_goal": "show crowded bag",
                "initial_state": "bag is crowded",
                "action": "hand moves a book",
                "final_state": "desk space appears",
                "camera_motion": "static",
                "visual_description": "crowded commuter bag",
                "subtitle": "leave room for essentials",
                "voiceover": "leave room for essentials",
                "asset_requirement": "none",
                "render_strategy": "text_to_video",
                "product_presence": "forbidden",
                "identity_strictness": "low",
                "review_focus": ["story"],
                "continuity_group": "desk_story",
                "transition_type": "hard_cut"
              },
              {
                "shot_index": 2,
                "duration_seconds": 3,
                "purpose": "continue the same desk scene",
                "narrative_role": "context",
                "scene_goal": "finish clearing space",
                "initial_state": "same crowded bag",
                "action": "hand clears desk space",
                "final_state": "desk is ready",
                "camera_motion": "static",
                "visual_description": "same commuter desk",
                "subtitle": "make room for what matters",
                "voiceover": "make room for what matters",
                "asset_requirement": "none",
                "render_strategy": "text_to_video",
                "product_presence": "forbidden",
                "identity_strictness": "low",
                "review_focus": ["story"],
                "continuity_group": "desk_story",
                "transition_type": "continue_from_previous"
              }
            ]
            """,
            "error": None,
        }

    monkeypatch.setattr("agent.video_generation_workflow._call_text_llm", fake_text_llm)

    storyboard = plan_director_storyboard(
        product_context={"duration_seconds": 15, "product_identity_card": {}},
        script_plan={"rich_story_text": "A detailed commute story with desk actions and a clear ending."},
        asset_analysis={"assets": [], "asset_profiles": []},
    )

    assert captured_prompt["script_plan"]["rich_story_text"].startswith("A detailed commute story")
    assert storyboard[1]["continuity_group"] == "desk_story"
    assert storyboard[1]["transition_type"] == "continue_from_previous"


def test_product_context_contains_shared_visual_style_bible():
    context = build_product_context(
        task_data={
            "title": "轻薄笔记本",
            "style": "product_showcase",
            "custom_style_prompt": "自然光，干净桌面",
            "duration_seconds": 15,
        },
        asset_analysis={"assets": [], "asset_profiles": [], "product_identity_card": {}},
    )

    assert context["visual_style_bible"]["realism"]
    assert "自然光" in context["visual_style_bible"]["lighting"]
    assert context["visual_style_bible"]["background_complexity"]


def test_normalize_storyboard_treats_null_continuity_group_as_empty():
    storyboard = _normalize_storyboard(
        [
            {
                "shot_index": 1,
                "duration_seconds": 3,
                "scene_goal": "show commuter bag",
                "continuity_group": None,
                "transition_type": None,
            }
        ]
    )

    assert storyboard[0]["continuity_group"] == ""
    assert storyboard[0]["transition_type"] == "hard_cut"


def test_ensure_storyboard_continuity_clears_tail_anchor_for_hard_cut():
    storyboard = _ensure_storyboard_continuity(
        [
            {
                "shot_index": 1,
                "transition_type": "hard_cut",
                "continuity_group": "",
                "anchor_last_frame": True,
            }
        ]
    )

    assert storyboard[0]["anchor_last_frame"] is False


def test_director_prompt_lists_state_transition_fields(monkeypatch):
    captured_prompt = {}

    def fake_text_llm(prompt, purpose, temperature):
        captured_prompt.update(prompt)
        return {"ok": True, "content": "[]", "error": None}

    monkeypatch.setattr("agent.video_generation_workflow._call_text_llm", fake_text_llm)

    plan_director_storyboard(
        product_context={"duration_seconds": 15, "product_identity_card": {}},
        script_plan={"rich_story_text": "detailed story"},
        asset_analysis={"assets": [], "asset_profiles": []},
    )

    output_format = captured_prompt["output_format"]
    assert "initial_state" in output_format
    assert "action" in output_format
    assert "final_state" in output_format
    assert "voiceover" in output_format
    assert "asset_requirement" in output_format


def test_normalize_storyboard_repairs_subtitle_voiceover_and_role_aliases():
    storyboard = _normalize_storyboard(
        [
            {
                "shot_index": 1,
                "duration_seconds": 3,
                "narrative_role": "proof",
                "scene_goal": "show stable material detail",
                "voiceover": "磨砂质感，一眼识别",
            },
            {
                "shot_index": 2,
                "duration_seconds": 3,
                "narrative_role": "cta",
                "scene_goal": "finish the story",
                "subtitle": "点击了解更多",
            },
            {
                "shot_index": 3,
                "duration_seconds": 3,
                "narrative_role": "product_reveal",
                "scene_goal": "轻薄机身自然登场",
            },
        ]
    )

    assert storyboard[0]["narrative_role"] == "detail_proof"
    assert storyboard[0]["subtitle"] == "磨砂质感，一眼识别"
    assert storyboard[1]["voiceover"] == "卖点结果看得见"
    assert storyboard[2]["subtitle"] == "轻薄机身自然登场"
    assert storyboard[2]["voiceover"] == "轻薄机身自然登场"


def test_normalize_storyboard_preserves_complete_public_subtitle():
    storyboard = _normalize_storyboard(
        [
            {
                "shot_index": 1,
                "duration_seconds": 3,
                "narrative_role": "feature_demo",
                "subtitle": "透明杯身一眼看清饮水余量",
                "voiceover": "透明杯身让饮水余量一眼看清，放在桌边也更安心。",
            }
        ]
    )

    assert storyboard[0]["subtitle"] == "透明杯身一眼看清饮水余量"
    assert storyboard[0]["voiceover"] == "透明杯身让饮水余量一眼看清，放在桌边也更安心"


def test_generic_fallback_copy_uses_product_context_without_fixed_cta_or_problem_solution():
    product_context = {
        "duration_seconds": 15,
        "title": "修护精华乳",
        "product_title": "修护精华乳",
        "selling_points": ["按压泵头定量取用", "夜间修护保湿", "磨砂玻璃质感"],
        "usage_scene": "晚间梳妆台护肤",
        "product_identity_card": {
            "product_type": "修护精华乳",
            "visible_marks": ["浅蓝标签", "银色泵头"],
            "key_components": ["银色泵头", "磨砂瓶身"],
            "material_features": ["磨砂玻璃质感"],
            "functional_features": ["按压泵头定量取用"],
            "must_preserve": ["浅蓝标签", "银色泵头"],
        },
    }

    script = _fallback_conservative_script(product_context, {})
    storyboard = _fallback_conservative_storyboard(
        product_context,
        {
            **script,
            "beats": script["beats"]
            + [
                {
                    "start_seconds": 15,
                    "end_seconds": 17,
                    "role": "cta",
                    "message": "结果收束",
                    "subtitle": "点击了解更多",
                }
            ],
        },
    )
    director_storyboard = _fallback_director_storyboard(
        product_context=product_context,
        script_plan={
            "hook": "看看这个好物",
            "body": product_context["selling_points"],
            "cta": "点击了解更多",
        },
        asset_analysis={
            "assets": [
                {
                    "asset_id": "asset-skincare",
                    "asset_type": "image",
                    "file_path": "/tmp/skincare.jpg",
                    "is_supported": True,
                }
            ],
            "asset_profiles": [],
        },
    )

    forbidden = {"看看这个好物", "点击了解更多", "点击查看详情", "真实外观确认", "真实体验"}
    visible_lines = set()
    visible_lines.update(str(script.get(key, "")).strip() for key in ("hook", "cta", "full_subtitle_script"))
    visible_lines.update(str(beat.get("subtitle", "")).strip() for beat in script.get("beats", []))
    visible_lines.update(str(shot.get("subtitle", "")).strip() for shot in storyboard + director_storyboard)
    visible_lines.update(str(shot.get("voiceover", "")).strip() for shot in storyboard + director_storyboard)

    assert not forbidden & visible_lines
    assert "按压泵头定量取用" in visible_lines
    assert "磨砂玻璃质感" in visible_lines
    assert script["beats"][0]["role"] == "hook"
    assert script["beats"][-1]["role"] == "cta"
    assert all(beat["role"] != "problem" for beat in script["beats"])
    assert director_storyboard[0]["narrative_role"] == "hook"


def test_normalize_storyboard_uses_short_caption_fallback_and_preserves_director_risk_fields():
    storyboard = _normalize_storyboard(
        [
            {
                "shot_index": 1,
                "duration_seconds": 2,
                "narrative_role": "product_reveal",
                "purpose": "作为 product_reveal 衔接镜，在统一棚拍背景上展示整机外观。",
                "product_presence": "optional",
                "identity_strictness": "medium",
            }
        ]
    )

    assert storyboard[0]["subtitle"] == "轻巧登场"
    assert storyboard[0]["voiceover"] == "轻巧登场"
    assert storyboard[0]["product_presence"] == "optional"
    assert storyboard[0]["identity_strictness"] == "medium"


def test_match_assets_does_not_bind_product_asset_to_forbidden_story_scene():
    matches = match_assets_to_storyboard(
        storyboard=[
            {
                "shot_index": 1,
                "narrative_role": "hook",
                "product_presence": "forbidden",
                "asset_id": "asset-product",
                "render_strategy": "image_to_video",
            }
        ],
        asset_analysis={
            "assets": [
                {
                    "asset_id": "asset-product",
                    "asset_type": "image",
                    "file_path": "/tmp/product.jpg",
                    "is_supported": True,
                }
            ]
        },
    )

    assert matches[0]["matched_asset"] is None
    assert matches[0]["strategy"] == "text_to_video"


def test_product_reveal_uses_complete_appearance_anchor_for_local_bridge():
    matches = match_assets_to_storyboard(
        storyboard=[
            {
                "shot_index": 1,
                "narrative_role": "product_reveal",
                "continuity_mode": "shared_scene_bridge",
                "product_presence": "optional",
            }
        ],
        asset_analysis={
            "shared_scene_background_path": "/tmp/studio_background.jpg",
            "assets": [
                {
                    "asset_id": "asset-detail",
                    "asset_type": "image",
                    "anchor_file_path": "/tmp/detail.jpg",
                    "file_path": "/tmp/detail.jpg",
                    "visual_role": "detail_reference",
                    "is_supported": True,
                },
                {
                    "asset_id": "asset-showcase",
                    "asset_type": "image",
                    "anchor_file_path": "/tmp/showcase.jpg",
                    "file_path": "/tmp/showcase.jpg",
                    "visual_role": "appearance_anchor",
                    "is_supported": True,
                },
            ],
        },
    )

    assert matches[0]["matched_asset"]["is_scene_background"] is True
    assert matches[0]["matched_asset"]["reveal_asset_path"] == "/tmp/showcase.jpg"


def test_laptop_product_prompt_forces_single_stable_device():
    prompt = _build_seedance_prompt(
        {
            "render_strategy": "image_to_video",
            "product_presence": "required",
            "identity_strictness": "high",
            "action": "打开笔记本并旋转展示",
            "asset": {"file_path": "/tmp/laptop.jpg"},
            "product_identity_card": {
                "product_type": "笔记本电脑",
                "appearance_summary": "黑色雷蛇笔记本电脑",
                "motion_affordance": {"forbidden_actions": ["不合理翻转", "悬浮", "变形"]},
            },
        }
    )

    assert "画面中只能保留一台商品主体" in prompt
    assert "不要打开、合上、折叠、翻转或改变铰链角度" in prompt


def test_forbidden_scene_prompt_does_not_repeat_product_description():
    prompt = _build_seedance_prompt(
        {
            "render_strategy": "text_to_video",
            "product_presence": "forbidden",
            "visual_description": "一台雷蛇笔记本悬浮翻转",
            "scene_goal": "建立性能问题",
        }
    )

    assert "雷蛇笔记本" not in prompt
    assert "不出现待售商品或同类商品" in prompt


def test_multimodal_asset_analysis_requests_json_object(monkeypatch):
    captured_payload = {}

    monkeypatch.delenv("AIGC_DISABLE_LLM", raising=False)
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    monkeypatch.setenv("ARK_TEXT_ENDPOINT_ID", "ep-test")
    monkeypatch.setattr(
        workflow,
        "_build_multimodal_image_batches",
        lambda image_paths: [["data:image/jpeg;base64,AAAA"]],
    )

    def fake_post(base_url, api_key, payload, purpose, timeout):
        captured_payload.update(payload)
        return {"ok": True, "content": "{}", "error": None}

    monkeypatch.setattr(workflow, "_post_openai_compatible_payload", fake_post)

    result = workflow._call_multimodal_llm(
        {"task": "analyze asset"},
        image_paths=["/tmp/product.jpg"],
        purpose="asset_analysis",
    )

    assert result["ok"] is True
    assert captured_payload["response_format"] == {"type": "json_object"}
    assert captured_payload["thinking"] == {"type": "disabled"}
    assert "纸质笔记本" in captured_payload["messages"][1]["content"][0]["text"]


def test_razer_notebook_title_is_disambiguated_as_laptop_computer():
    assert workflow._infer_product_type("雷蛇笔记本介绍") == "笔记本电脑"


def test_product_free_hook_prompt_forbids_identity_and_keeps_bridge_cue():
    prompt = _build_seedance_prompt(
        {
            "render_strategy": "text_to_video",
            "product_presence": "forbidden",
            "narrative_role": "hook",
            "product_identity_card": {"product_type": "笔记本电脑"},
        }
    )

    # 不把可识别商品/品牌画进铺垫镜
    assert "整理桌面上的书本" not in prompt
    assert "品牌 logo" in prompt
    assert "笔记本电脑" in prompt
    # 通用桥：无场景元素时回退到按品类插值的无品牌携带/使用情景，保留镜头目的
    assert "包袋" in prompt or "使用环境" in prompt


def test_content_repair_reuses_renderer_concat_instead_of_system_ffmpeg(tmp_path, monkeypatch):
    output_dir = tmp_path / "task"
    output_dir.mkdir()
    repaired_clip = output_dir / "seedance_shot_01_repaired.mp4"
    repaired_clip.write_bytes(b"repaired")
    # 本地承接镜是中间产物，不能被重新拼接逻辑误认为正式分镜。
    (output_dir / "seedance_shot_02_local_scene.mp4").write_bytes(b"local-scene")

    monkeypatch.setattr(
        workflow,
        "repair_and_rerender_shot",
        lambda **kwargs: {"success": True, "clip_path": str(repaired_clip)},
    )

    def fail_if_system_ffmpeg_is_called(*args, **kwargs):
        raise AssertionError("局部修复不能直接依赖系统 PATH 中的 ffmpeg")

    monkeypatch.setattr("subprocess.run", fail_if_system_ffmpeg_is_called)

    concat_calls = []

    def fake_concat(clip_paths, final_video_path, transition_types=None):
        concat_calls.append((clip_paths, transition_types))
        final_video_path.write_bytes(b"raw")
        return {"success": True, "error": None}

    def fake_overlay(source_video_path, final_video_path, shots):
        final_video_path.write_bytes(source_video_path.read_bytes())
        return {"success": True, "error": None}

    monkeypatch.setattr("agent.seedance_video_renderer._concat_videos", fake_concat)
    monkeypatch.setattr("agent.seedance_video_renderer._overlay_storyboard_subtitles", fake_overlay)

    workflow._repair_rendered_content(
        task_id="task-test",
        repair_records=[{"shot_index": 1, "repair_strategy": "rerender_with_stronger_identity_anchor"}],
        creation_plan={
            "shots": [
                {
                    "shot_index": 1,
                    "transition_type": "hard_cut",
                    "render_strategy": "image_to_video",
                    "product_presence": "required",
                }
            ]
        },
        render_result={"shot_results": [{"shot_index": 1}]},
        output_dir=str(output_dir),
        report=lambda *args, **kwargs: None,
    )

    assert concat_calls
    assert [path.name for path in concat_calls[0][0]] == ["seedance_shot_01.mp4"]
    assert (output_dir / "seedance_final.mp4").read_bytes() == b"raw"


def test_content_repair_skips_identity_anchor_rerender_for_forbidden_story_scene(tmp_path, monkeypatch):
    rerendered_shots = []

    def fake_repair(**kwargs):
        rerendered_shots.append(kwargs["shot_index"])
        return {"success": False, "error": "not expected"}

    monkeypatch.setattr(workflow, "repair_and_rerender_shot", fake_repair)

    workflow._repair_rendered_content(
        task_id="task-test",
        repair_records=[{"shot_index": 1, "repair_strategy": "rerender_with_stronger_identity_anchor"}],
        creation_plan={
            "shots": [
                {
                    "shot_index": 1,
                    "render_strategy": "text_to_video",
                    "product_presence": "forbidden",
                }
            ]
        },
        render_result={"shot_results": [{"shot_index": 1}]},
        output_dir=str(tmp_path),
        report=lambda *args, **kwargs: None,
    )

    assert rerendered_shots == []


def test_normalize_script_plan_preserves_executable_beat_contract():
    script = _normalize_script_plan(
        {
            "rich_story_text": "通勤者先整理拥挤书桌，再让轻薄笔记本自然出现，最后稳定收束到真实商品外观。",
            "beats": [
                {
                    "start_seconds": 0,
                    "end_seconds": 3,
                    "role": "problem",
                    "message": "展示拥挤桌面带来的通勤压力。",
                    "subtitle": "桌面总是不够用",
                    "visual_intent": "书本和背包占满桌面。",
                    "scene_before": "桌面被书本和背包占满。",
                    "action": "一只手把厚重书本移到背包旁。",
                    "scene_after": "桌面中央腾出一块空间。",
                    "visible_entities": ["书本", "背包", "一只手"],
                    "physical_constraints": ["不出现待售笔记本电脑", "书本只能由手移动"],
                    "asset_requirements": ["不需要真实商品素材"],
                    "continuity_mode": "new_scene",
                    "transition_reason": "用通勤桌面问题建立使用场景。",
                }
            ],
        },
        {"duration_seconds": 15},
    )

    assert script is not None
    beat = script["beats"][0]
    assert beat["scene_before"] == "桌面被书本和背包占满。"
    assert beat["action"] == "一只手把厚重书本移到背包旁。"
    assert beat["scene_after"] == "桌面中央腾出一块空间。"
    assert beat["visible_entities"] == ["书本", "背包", "一只手"]
    assert beat["physical_constraints"] == ["不出现待售笔记本电脑", "书本只能由手移动"]
    assert beat["asset_requirements"] == ["不需要真实商品素材"]
    assert beat["continuity_mode"] == "new_scene"
    assert beat["transition_reason"] == "用通勤桌面问题建立使用场景。"


def test_script_review_rejects_beats_without_executable_scene_contract():
    review = review_script_plan(
        {
            "hook": "背包不该被电脑占满",
            "body": ["轻薄机身", "通勤更轻松"],
            "cta": "点击了解更多",
            "tone": "写实通勤场景",
            "rich_story_text": (
                "通勤者把背包放到桌面，准备整理第二天需要携带的东西。"
                "视频中段让轻薄笔记本自然出现，借助真实外观说明它可以减少通勤负担。"
                "结尾回到稳定的商品全景，让用户理解轻薄带来的实际价值并点击了解。"
            ),
            "beats": [
                {"role": "problem", "visual_intent": "拥挤桌面。"},
                {"role": "feature_demo", "visual_intent": "展示笔记本。"},
                {"role": "cta", "visual_intent": "商品全景收束。"},
            ],
        }
    )

    assert review["passed"] is False
    assert any("scene_before" in issue for issue in review["issues"])
    assert any("visible_entities" in issue for issue in review["issues"])
    assert any("physical_constraints" in issue for issue in review["issues"])


def test_script_review_requires_transition_reason_for_new_scene_after_first_beat():
    common_contract = {
        "visible_entities": ["桌面", "商品"],
        "physical_constraints": ["商品外观保持稳定"],
        "asset_requirements": ["使用真实商品外观锚点"],
    }
    review = review_script_plan(
        {
            "hook": "轻一点，通勤就从容一点",
            "body": ["轻薄机身", "真实外观"],
            "cta": "点击了解更多",
            "tone": "自然写实",
            "rich_story_text": (
                "视频先展示桌面空间不足的问题，再切到同一环境下的轻薄笔记本。"
                "中段用稳定的商品全景证明轻便价值，最后保留完整商品画面并给出行动引导。"
            ),
            "beats": [
                {
                    **common_contract,
                    "role": "problem",
                    "visual_intent": "拥挤桌面。",
                    "scene_before": "桌面被书本占满。",
                    "action": "手把书本移开。",
                    "scene_after": "桌面中央腾出空间。",
                    "continuity_mode": "new_scene",
                    "transition_reason": "建立问题场景。",
                },
                {
                    **common_contract,
                    "role": "feature_demo",
                    "visual_intent": "商品进入桌面。",
                    "scene_before": "桌面中央已经腾出空间。",
                    "action": "商品稳定出现在桌面中央。",
                    "scene_after": "商品完整可见。",
                    "continuity_mode": "new_scene",
                    "transition_reason": "",
                },
                {
                    **common_contract,
                    "role": "cta",
                    "visual_intent": "稳定收束。",
                    "scene_before": "商品完整可见。",
                    "action": "镜头保持稳定并为字幕预留空间。",
                    "scene_after": "商品全景稳定收束。",
                    "continuity_mode": "continue_previous",
                    "transition_reason": "",
                },
            ],
        }
    )

    assert review["passed"] is False
    assert any("transition_reason" in issue for issue in review["issues"])


def test_script_prompt_requests_executable_scene_contract(monkeypatch):
    captured_prompt = {}

    def fake_text_llm(prompt, purpose, temperature):
        captured_prompt.update(prompt)
        return {"ok": False, "content": "", "error": "disabled"}

    monkeypatch.setattr("agent.video_generation_workflow._call_text_llm", fake_text_llm)

    plan_script(
        product_context={
            "duration_seconds": 15,
            "product_title": "轻薄笔记本",
            "selling_points": ["方便通勤"],
            "product_identity_card": {"appearance_summary": "黑色磨砂笔记本"},
        },
        director_decision={},
    )

    beat_format = captured_prompt["output_format"]["beats"][0]
    assert "scene_before" in beat_format
    assert "action" in beat_format
    assert "scene_after" in beat_format
    assert "visible_entities" in beat_format
    assert "physical_constraints" in beat_format
    assert "asset_requirements" in beat_format
    assert "continuity_mode" in beat_format
    assert "transition_reason" in beat_format


def test_normalize_script_plan_preserves_paper_style_context_and_shot_fields():
    script = _normalize_script_plan(
        {
            "context_reconstruction": {
                "scene_setting": "自然光办公桌面。",
                "plot_development": "先建立桌面拥挤问题，再展示轻薄电脑。",
                "emotional_tendency": "从压迫转为轻松。",
                "speaking_intent": "用真实场景说明便携价值。",
                "causal_chain": ["桌面拥挤", "腾出空间", "电脑自然出现"],
            },
            "beats": [
                {
                    "start_seconds": 0,
                    "end_seconds": 3,
                    "role": "problem",
                    "message": "建立桌面拥挤问题。",
                    "subtitle": "桌面总是不够用",
                    "visual_intent": "手移开桌面杂物。",
                    "scene_before": "桌面被杂物占满。",
                    "action": "手把水杯和充电线移到侧边。",
                    "scene_after": "桌面中央腾出空间。",
                    "visible_entities": ["桌面", "水杯", "充电线", "一只手"],
                    "physical_constraints": ["道具只能由手移动"],
                    "asset_requirements": ["不需要真实商品素材"],
                    "continuity_mode": "new_scene",
                    "transition_reason": "建立用户问题。",
                    "shot_type": "中景",
                    "camera_movement": "固定镜头",
                    "scene_description": "自然光桌面，画面克制写实。",
                    "subject_appearance": "一只手和普通无品牌桌面道具。",
                    "subject_position": "水杯位于画面左侧，充电线位于桌面中央。",
                    "acting_direction": "手从画面右侧进入并缓慢整理道具。",
                    "dialogue": "[No Dialogue]",
                    "scene_elements": ["木质桌面", "水杯", "充电线"],
                    "cut_reason": "空间整理完成后切入商品揭示。",
                }
            ],
        },
        {"duration_seconds": 15},
    )

    assert script is not None
    assert script["context_reconstruction"]["causal_chain"] == ["桌面拥挤", "腾出空间", "电脑自然出现"]
    beat = script["beats"][0]
    assert beat["shot_type"] == "中景"
    assert beat["camera_movement"] == "固定镜头"
    assert beat["subject_position"].startswith("水杯位于")
    assert beat["dialogue"] == "[No Dialogue]"
    assert beat["scene_elements"] == ["木质桌面", "水杯", "充电线"]
    assert beat["cut_reason"] == "空间整理完成后切入商品揭示。"


def test_script_review_rejects_missing_paper_style_verification_fields():
    common = {
        "visual_intent": "稳定可拍的镜头。",
        "scene_before": "桌面保持稳定。",
        "action": "手完成一个简单动作。",
        "scene_after": "动作完成后桌面保持稳定。",
        "visible_entities": ["桌面", "一只手"],
        "physical_constraints": ["动作符合真实物理规律"],
        "asset_requirements": ["按需使用真实素材"],
        "continuity_mode": "continue_previous",
        "transition_reason": "",
    }
    review = review_script_plan(
        {
            "hook": "轻一点，通勤更轻松",
            "body": ["轻薄机身", "真实外观"],
            "cta": "点击了解更多",
            "tone": "自然写实",
            "rich_story_text": (
                "视频先通过真实桌面建立通勤负担，再让商品自然进入同一空间。"
                "随后用稳定商品画面展示外观和卖点，最后完整收束并给出行动引导。"
            ),
            "context_reconstruction": {},
            "beats": [
                {**common, "role": "problem", "continuity_mode": "new_scene", "transition_reason": "建立场景。"},
                {**common, "role": "feature_demo"},
                {**common, "role": "cta"},
            ],
        }
    )

    assert review["passed"] is False
    assert any("context_reconstruction.scene_setting" in issue for issue in review["issues"])
    assert any("shot_type" in issue for issue in review["issues"])
    assert any("dialogue" in issue for issue in review["issues"])
    assert any("subject_position" in issue for issue in review["issues"])
    assert any("cut_reason" in issue for issue in review["issues"])


def test_script_review_rejects_continued_scene_without_shared_elements(monkeypatch):
    monkeypatch.setattr(
        "agent.video_generation_workflow._call_text_llm",
        lambda *args, **kwargs: {"ok": False, "content": "", "error": "disabled"},
    )
    script = plan_script(
        product_context={
            "duration_seconds": 15,
            "product_title": "轻薄笔记本电脑",
            "selling_points": ["方便通勤"],
            "product_identity_card": {"appearance_summary": "黑色磨砂笔记本电脑"},
        },
        director_decision={},
    )
    script["beats"][2]["scene_elements"] = ["完全不同的户外场景"]

    review = review_script_plan(script)

    assert review["passed"] is False
    assert any("延续上一场景" in issue and "scene_elements" in issue for issue in review["issues"])


def test_script_prompt_requests_paper_style_three_stage_contract(monkeypatch):
    captured_prompt = {}

    def fake_text_llm(prompt, purpose, temperature):
        captured_prompt.update(prompt)
        return {"ok": False, "content": "", "error": "disabled"}

    monkeypatch.setattr("agent.video_generation_workflow._call_text_llm", fake_text_llm)

    plan_script(
        product_context={
            "duration_seconds": 15,
            "product_title": "轻薄笔记本电脑",
            "selling_points": ["方便通勤"],
            "product_identity_card": {"appearance_summary": "黑色磨砂笔记本电脑"},
        },
        director_decision={},
    )

    assert "context_reconstruction" in captured_prompt["output_format"]
    beat_format = captured_prompt["output_format"]["beats"][0]
    for field in (
        "shot_type",
        "camera_movement",
        "scene_description",
        "subject_appearance",
        "subject_position",
        "acting_direction",
        "dialogue",
        "scene_elements",
        "cut_reason",
    ):
        assert field in beat_format
    assert "dialogue_completeness" in captured_prompt["script_verification_modules"]
    assert "scene_coherence" in captured_prompt["script_verification_modules"]
    assert "positional_physical_rationality" in captured_prompt["script_verification_modules"]


def test_text_llm_requests_json_object(monkeypatch):
    captured_payload = {}

    def fake_post(base_url, api_key, payload, purpose, timeout):
        captured_payload.update(payload)
        return {"ok": True, "content": "{}", "error": None}

    monkeypatch.setattr(workflow, "_post_openai_compatible_payload", fake_post)

    result = workflow._call_openai_compatible_api(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model="deepseek-chat",
        prompt_data={"task": "return json"},
        purpose="script_plan",
    )

    assert result["ok"] is True
    assert captured_payload["response_format"] == {"type": "json_object"}


def test_script_prompt_includes_product_identity_grounding_contract(monkeypatch):
    captured_prompt = {}

    def fake_text_llm(prompt, purpose, temperature):
        captured_prompt.update(prompt)
        return {"ok": False, "content": "", "error": "disabled"}

    monkeypatch.setattr("agent.video_generation_workflow._call_text_llm", fake_text_llm)

    plan_script(
        product_context={
            "duration_seconds": 15,
            "product_title": "雷蛇灵刃16游戏笔记本电脑",
            "selling_points": ["轻薄便携"],
            "product_identity_card": {
                "product_type": "笔记本电脑",
                "appearance_summary": "黑色磨砂机身，A面中央有绿色标识。",
            },
        },
        director_decision={},
    )

    grounding = captured_prompt["product_grounding_contract"]
    assert grounding["expected_product_type"] == "笔记本电脑"
    assert "不得改写成其他商品" in grounding["hard_rule"]
    assert "grounded_product_type" in captured_prompt["output_format"]


def test_script_review_rejects_grounded_product_type_mismatch(monkeypatch):
    monkeypatch.setattr(
        "agent.video_generation_workflow._call_text_llm",
        lambda *args, **kwargs: {"ok": False, "content": "", "error": "disabled"},
    )
    script = plan_script(
        product_context={
            "duration_seconds": 15,
            "product_title": "雷蛇灵刃16游戏笔记本电脑",
            "selling_points": ["轻薄便携"],
            "product_identity_card": {"product_type": "笔记本电脑"},
        },
        director_decision={},
    )
    script["grounded_product_type"] = "保温杯"

    review = review_script_plan(script)

    assert review["passed"] is False
    assert any("grounded_product_type" in issue for issue in review["issues"])


def test_creative_text_tasks_prefer_ark_before_deepseek(monkeypatch):
    called_models = []

    monkeypatch.setenv("AIGC_DISABLE_LLM", "0")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-key")
    monkeypatch.setenv("ARK_API_KEY", "ark-key")
    monkeypatch.setenv("ARK_TEXT_ENDPOINT_ID", "ep-ark-text")
    monkeypatch.delenv("TEXT_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("TEXT_LLM_MODEL", raising=False)

    def fake_call(base_url, api_key, model, prompt_data, purpose, temperature, timeout):
        called_models.append(model)
        return {"ok": True, "content": "{}", "error": None}

    monkeypatch.setattr(workflow, "_call_openai_compatible_api", fake_call)

    result = workflow._call_text_llm({"task": "return json"}, purpose="script_plan")

    assert result["ok"] is True
    assert called_models == ["ep-ark-text"]


def test_first_required_product_shot_uses_full_uploaded_frame_even_when_story_starts_text(tmp_path):
    logo_anchor = tmp_path / "logo_anchor.jpg"
    full_frame = tmp_path / "full_laptop_frame.jpg"
    logo_anchor.write_bytes(b"logo")
    full_frame.write_bytes(b"full laptop")

    storyboard = [
        {
            "shot_index": 1,
            "duration_seconds": 3,
            "narrative_role": "hook",
            "render_strategy": "text_to_video",
            "product_presence": "forbidden",
            "identity_strictness": "low",
            "scene_goal": "湖面移动",
            "visual_description": "湖面移动",
        },
        {
            "shot_index": 2,
            "duration_seconds": 4,
            "narrative_role": "feature_demo",
            "render_strategy": "image_to_video",
            "product_presence": "required",
            "identity_strictness": "high",
        },
    ]
    asset_analysis = {
        "assets": [
            {
                "asset_id": "asset_laptop",
                "asset_type": "image",
                "is_supported": True,
                "file_path": str(logo_anchor),
                "standardized_file_path": str(full_frame),
                "original_file_path": str(full_frame),
                "anchor_file_path": str(logo_anchor),
                "visual_role": "appearance_anchor",
                "quality_score": 90,
                "keyframe_variants": {"hero": str(logo_anchor), "detail": str(logo_anchor)},
            }
        ]
    }

    matches = match_assets_to_storyboard(storyboard, asset_analysis)
    plan = build_creation_plan(
        {"target_platform": "tiktok", "product_identity_card": {"must_preserve": ["笔记本整机"]}},
        storyboard,
        matches,
    )

    story = plan["shots"][0]
    assert story["shot_index"] == 1
    assert story["render_strategy"] == "text_to_video"
    assert story["product_presence"] == "forbidden"
    assert story["force_full_frame_anchor"] is False
    assert "湖面" in story["visual_description"]

    product = plan["shots"][1]
    assert product["shot_index"] == 2
    assert product["render_strategy"] == "image_to_video"
    assert product["product_presence"] == "required"
    assert product["identity_strictness"] == "high"
    assert product["force_full_frame_anchor"] is True
    assert product["render_input"]["file_path"] == str(full_frame)
    assert "湖面" not in product["visual_description"]


def test_build_creation_plan_preserves_material_aware_first_product_action(tmp_path):
    anchor = tmp_path / "cup_full_frame.jpg"
    anchor.write_bytes(b"cup")
    storyboard = [
        {
            "shot_index": 1,
            "duration_seconds": 5,
            "narrative_role": "product_reveal",
            "render_strategy": "image_to_video",
            "product_presence": "required",
            "identity_strictness": "high",
            "scene_goal": "从素材原场景建立真实商品身份",
            "initial_state": "第一帧是上传素材里的水杯和桌面。",
            "action": "由 LLM 根据素材首帧、商品结构、卖点和 skill 样例决定本镜头动作。",
            "visual_description": "第一帧承接上传素材中的真实水杯和原有桌面环境，沿同一地点自然延展，不换地点，不新增第二个商品。",
            "subtitle": "通勤随手带",
            "voiceover": "通勤随手带",
            "asset_id": "cup_anchor",
            "material_strategy": "source_scene_extension",
            "selected_prompt_skill": "commerce_scene.source_confirm",
            "asset_usage_reason": "整杯完整清楚，适合从素材场景延展。",
            "planner_source": "product_fidelity_v3_skill_guided:material_first_source_scene_extension",
        },
        {
            "shot_index": 2,
            "duration_seconds": 5,
            "narrative_role": "usage_context",
            "render_strategy": "text_to_video",
            "product_presence": "forbidden",
            "scene_goal": "独立生活场景承接卖点",
            "visual_description": "真实通勤场景，不出现待售水杯。",
            "subtitle": "出门更省心",
            "voiceover": "出门更省心",
        },
    ]
    matches = [
        {
            "shot_index": 1,
            "strategy": "image_to_video",
            "matched_asset": {
                "asset_id": "cup_anchor",
                "asset_type": "image",
                "file_path": str(anchor),
                "standardized_file_path": str(anchor),
                "visual_role": "appearance_anchor",
            },
            "source_asset": {
                "asset_id": "cup_anchor",
                "asset_type": "image",
                "file_path": str(anchor),
                "standardized_file_path": str(anchor),
                "visual_role": "appearance_anchor",
            },
            "render_input": {
                "type": "asset",
                "asset_id": "cup_anchor",
                "file_path": str(anchor),
                "render_mode": "image_to_video",
            },
            "reference_scope": "full_product",
        },
        {
            "shot_index": 2,
            "strategy": "text_to_video",
            "matched_asset": None,
        },
    ]

    plan = build_creation_plan(
        {"target_platform": "tiktok", "product_identity_card": {"must_preserve": ["水杯外观"]}},
        storyboard,
        matches,
    )

    shot = plan["shots"][0]
    assert "短距离拿起" not in shot["action"]
    assert "LLM" in shot["action"] or "skill" in shot["action"]
    assert "只允许轻微镜头推进" not in shot["action"]
    assert shot["material_strategy"] == "source_scene_extension"
    assert shot["selected_prompt_skill"] == "commerce_scene.source_confirm"
    assert shot["asset_usage_reason"] == "整杯完整清楚，适合从素材场景延展。"


def _fake_preprocessed_image_asset(monkeypatch, tmp_path, *, asset_id="asset-product"):
    image_path = tmp_path / f"{asset_id}.jpg"
    Image.new("RGB", (80, 80), (230, 230, 230)).save(image_path)

    monkeypatch.setattr(
        workflow,
        "preprocess_all_assets",
        lambda assets, output_dir: [{
            "original_path": str(image_path),
            "output_path": str(image_path),
            "anchor_output_path": str(image_path),
            "keyframe_variants": {},
            "primary_product": {},
            "background_removed": False,
            "sharpness_fixed": False,
            "exposure_fixed": False,
        }],
    )
    monkeypatch.setattr(workflow, "create_studio_background", lambda output_dir: str(tmp_path / "studio.jpg"))

    return {
        "asset_id": asset_id,
        "asset_type": "image",
        "file_path": str(image_path),
    }


def _raw_response_text(result):
    raw_response = result.get("raw_response")
    if isinstance(raw_response, str):
        return raw_response
    if isinstance(raw_response, dict):
        return "\n".join(str(value) for value in raw_response.values())
    return ""


def _material_role_names(record):
    names = set()
    for key in ("normalized_roles", "material_capabilities"):
        value = record.get(key)
        if isinstance(value, list):
            names.update(str(item) for item in value)
        elif isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, bool) and nested_value:
                    names.add(str(nested_key))
                elif isinstance(nested_value, list):
                    names.update(str(item) for item in nested_value)
    return names


def test_process_assets_records_multimodal_parse_failure_and_repair_metadata(monkeypatch, tmp_path):
    uploaded_asset = _fake_preprocessed_image_asset(monkeypatch, tmp_path, asset_id="asset-cup")
    raw_bad_response = "模型说明：这是一只水杯，但我没有按 JSON 返回。"
    repaired_response = {
        "asset_roles": [
            {
                "asset_id": "asset-cup",
                "suitable_for": ["product_showcase"],
                "visual_role": "appearance_anchor",
                "reason": "主体完整清楚，适合作为商品外观锚点。",
                "quality_score": 86,
                "product_visibility": "主体清晰",
                "background_type": "干净背景",
                "identity_contribution": ["杯身形状", "杯盖颜色"],
                "risk_notes": [],
            }
        ],
        "product_identity_card": {
            "product_type": "水杯",
            "identity_confidence": "high",
            "appearance_summary": "透明棕色杯身，黄色杯盖，蓝色环。",
            "must_preserve": ["透明棕色杯身", "黄色杯盖", "蓝色环"],
            "reference_asset_ids": ["asset-cup"],
            "motion_affordance": {
                "can_be_handheld": True,
                "allowed_actions": ["手持携带", "放在通勤场景中"],
                "forbidden_actions": [],
            },
        },
    }

    monkeypatch.setattr(
        workflow,
        "_call_multimodal_llm",
        lambda *args, **kwargs: {"ok": True, "content": raw_bad_response, "error": None},
    )
    monkeypatch.setattr(
        workflow,
        "_call_text_llm",
        lambda *args, **kwargs: {"ok": True, "content": workflow.json.dumps(repaired_response), "error": None},
    )

    result = workflow.process_assets({
        "task_id": "task-vision-repair",
        "title": "通勤水杯",
        "selling_points": ["大容量", "便携"],
        "uploaded_assets": [uploaded_asset],
    })

    assert result["vision_parse_failed"] is True
    assert result["vision_parse_repaired"] is True
    assert result["fallback_used"] is False
    assert result.get("fallback_reason") in ("", None)
    assert raw_bad_response in _raw_response_text(result)
    assert result["product_identity_card"]["llm_enabled"] is True
    assert result["product_identity_card"]["identity_confidence"] == "high"
    assert result["asset_profiles"][0]["asset_id"] == "asset-cup"


def test_process_assets_marks_fallback_when_multimodal_parse_repair_fails(monkeypatch, tmp_path):
    uploaded_asset = _fake_preprocessed_image_asset(monkeypatch, tmp_path, asset_id="asset-laptop")
    raw_bad_response = "不是 JSON，也没有可修复的对象。"

    monkeypatch.setattr(
        workflow,
        "_call_multimodal_llm",
        lambda *args, **kwargs: {"ok": True, "content": raw_bad_response, "error": None},
    )
    monkeypatch.setattr(
        workflow,
        "_call_text_llm",
        lambda *args, **kwargs: {"ok": False, "content": "", "error": "repair failed"},
    )

    result = workflow.process_assets({
        "task_id": "task-vision-fallback",
        "title": "雷蛇笔记本电脑",
        "selling_points": ["轻薄", "高性能"],
        "uploaded_assets": [uploaded_asset],
    })

    assert result["vision_parse_failed"] is True
    assert result["vision_parse_repaired"] is False
    assert result["fallback_used"] is True
    assert "多模态" in result["fallback_reason"] or "parse" in result["fallback_reason"].lower()
    assert raw_bad_response in _raw_response_text(result)
    assert result["product_identity_card"]["llm_enabled"] is False
    assert result["product_identity_card"]["identity_confidence"] == "low"


def test_scene_context_product_showcase_asset_becomes_appearance_anchor_candidate(monkeypatch, tmp_path):
    uploaded_asset = _fake_preprocessed_image_asset(monkeypatch, tmp_path, asset_id="asset-scene-cup")
    multimodal_response = {
        "asset_roles": [
            {
                "asset_id": "asset-scene-cup",
                "suitable_for": ["scene_context", "product_showcase"],
                "visual_role": "scene_context",
                "reason": "水杯在真实桌面场景中，商品主体完整清楚，既有场景也能约束外观。",
                "quality_score": 82,
                "product_visibility": "主体清晰",
                "background_type": "场景背景",
                "identity_contribution": ["杯身比例", "杯盖和吸管颜色"],
                "risk_notes": [],
            }
        ],
        "product_identity_card": {
            "product_type": "水杯",
            "identity_confidence": "high",
            "appearance_summary": "透明棕色杯身，蓝色环，黄色杯盖和吸管细节。",
            "must_preserve": ["透明棕色杯身", "蓝色环", "黄色杯盖"],
            "reference_asset_ids": ["asset-scene-cup"],
            "motion_affordance": {"can_be_handheld": True, "allowed_actions": [], "forbidden_actions": []},
        },
    }

    monkeypatch.setattr(
        workflow,
        "_call_multimodal_llm",
        lambda *args, **kwargs: {
            "ok": True,
            "content": workflow.json.dumps(multimodal_response, ensure_ascii=False),
            "error": None,
        },
    )

    result = workflow.process_assets({
        "task_id": "task-role-normalization",
        "title": "水杯",
        "selling_points": ["通勤便携"],
        "uploaded_assets": [uploaded_asset],
    })

    profile = result["asset_profiles"][0]
    asset = result["assets"][0]
    assert profile["visual_role"] == "scene_context"
    assert "product_showcase" in profile["suitable_for"]
    assert profile["product_visibility"] == "主体清晰"
    assert (
        "appearance_anchor_candidate" in _material_role_names(profile)
        or "appearance_anchor_candidate" in _material_role_names(asset)
    )


def test_workflow_keeps_default_render_and_adds_ideal_commerce_ab_variant(monkeypatch, tmp_path):
    image_path = tmp_path / "product.jpg"
    image_path.write_bytes(b"image")
    render_calls = []

    monkeypatch.setattr(workflow, "_load_local_env", lambda: None)
    monkeypatch.setattr(
        workflow,
        "process_assets",
        lambda task_data: {
            "assets": [{
                "asset_id": "asset-product",
                "asset_type": "image",
                "is_supported": True,
                "file_path": str(image_path),
                "visual_role": "appearance_anchor",
            }],
            "asset_profiles": [{
                "asset_id": "asset-product",
                "visual_role": "appearance_anchor",
                "quality_score": 90,
                "product_visibility": "主体清晰",
            }],
            "product_identity_card": {
                "product_type": "水杯",
                "identity_confidence": "high",
                "appearance_anchor_available": True,
                "must_preserve": ["水杯外观"],
            },
            "shared_scene_background_path": "",
        },
    )
    monkeypatch.setattr(
        workflow,
        "structurize_user_requirements",
        lambda task_data: {
            "target_audience": "通勤人群",
            "usage_scene": "上班通勤",
            "creative_goal": "展示便携和大容量",
            "selling_point_priority": ["便携", "大容量"],
            "must_preserve": ["商品外观"],
            "avoid": [],
            "tone": "",
            "extra_requirements": "",
            "input_confidence": "medium",
        },
    )
    monkeypatch.setattr(
        workflow,
        "build_product_context",
        lambda task_data, asset_analysis: {
            "duration_seconds": 15,
            "target_platform": "tiktok",
            "product_title": task_data["title"],
            "selling_points": task_data["selling_points"],
            "product_identity_card": asset_analysis["product_identity_card"],
            "asset_profiles": asset_analysis["asset_profiles"],
        },
    )
    monkeypatch.setattr(workflow, "build_asset_capability_plan", lambda asset_analysis, product_context: {})
    monkeypatch.setattr(
        workflow,
        "plan_storyboard_from_template",
        lambda product_context, asset_analysis: (
            [{
                "shot_index": 1,
                "duration_seconds": 3,
                "narrative_role": "product_reveal",
                "render_strategy": "image_to_video",
                "product_presence": "required",
                "visual_description": "从素材图确认商品身份。",
                "asset_id": "asset-product",
            }],
            {"beats": [{"beat_index": 1, "goal": "确认商品"}]},
        ),
    )
    monkeypatch.setattr(workflow, "review_storyboard_shootability", lambda *args, **kwargs: {"passed": True, "issues": []})
    monkeypatch.setattr(workflow, "_enforce_storyboard_continuity_groups", lambda storyboard: storyboard)
    monkeypatch.setattr(workflow, "_rule_based_narrative_review", lambda *args, **kwargs: {"passed": True, "issues": []})
    monkeypatch.setattr(workflow, "match_assets_to_storyboard", lambda storyboard, asset_analysis: [])
    monkeypatch.setattr(
        workflow,
        "complete_asset_gaps",
        lambda storyboard, asset_matching, asset_analysis, product_identity_card: {"asset_matching": asset_matching},
    )
    monkeypatch.setattr(
        workflow,
        "build_creation_plan",
        lambda product_context, storyboard, asset_matching: {
            "total_duration_seconds": sum(shot.get("duration_seconds", 0) for shot in storyboard),
            "shots": storyboard,
            "variant_strategy": "A_conservative_fidelity",
        },
    )

    def fake_render_seedance_video(task_id, creation_plan, output_dir):
        output_dir = str(output_dir)
        render_calls.append({"task_id": task_id, "creation_plan": creation_plan, "output_dir": output_dir})
        variant = (
            "B_ideal_commerce_scene"
            if "variants/B_ideal_commerce_scene" in output_dir.replace("\\", "/")
            else "A_conservative_fidelity"
        )
        return {
            "success": True,
            "variant": variant,
            "video_path": str(Path(output_dir) / "seedance_final.mp4"),
            "shot_results": [],
        }

    monkeypatch.setattr(workflow, "render_seedance_video", fake_render_seedance_video)
    monkeypatch.setattr(
        workflow,
        "review_rendered_video_content",
        lambda *args, **kwargs: {"passed": True, "issues": [], "shot_reviews": []},
    )
    monkeypatch.setattr(workflow, "run_final_check", lambda *args, **kwargs: {"passed": True, "issues": []})
    monkeypatch.setattr(workflow, "_save_workflow_artifacts", lambda task_id, output_dir, artifacts: str(tmp_path / "artifacts"))

    result = workflow.run_video_generation_workflow({
        "task_id": "task-ab-contract",
        "title": "水杯",
        "selling_points": ["便携", "大容量"],
        "target_platform": "tiktok",
        "duration_seconds": 15,
        "uploaded_assets": [{"asset_id": "asset-product", "asset_type": "image", "file_path": str(image_path)}],
    })

    assert result["render_result"]["variant"] == "A_conservative_fidelity"
    assert result["render_result"]["video_path"] == str(tmp_path / "seedance_final.mp4")
    assert "B_ideal_commerce_scene" in result["ab_variants"]
    b_variant = result["ab_variants"]["B_ideal_commerce_scene"]
    assert b_variant["render_result"]["variant"] == "B_ideal_commerce_scene"
    assert b_variant["video_path"] == str(tmp_path / "variants" / "B_ideal_commerce_scene" / "seedance_final.mp4")
    assert len(render_calls) >= 2
    assert render_calls[0]["output_dir"] == str(tmp_path)
    assert any(
        "variants/B_ideal_commerce_scene" in call["output_dir"].replace("\\", "/")
        for call in render_calls[1:]
    )


def test_workflow_can_pause_after_script_review_without_rendering(monkeypatch, tmp_path):
    render_called = False
    storyboard = [
        {
            "shot_index": 1,
            "duration_seconds": 4,
            "narrative_role": "hook",
            "scene_goal": "展示通勤痛点",
            "initial_state": "杯子放在拥挤桌面上",
            "action": "用户把杯子装入包侧袋",
            "final_state": "杯子稳定露出",
            "subtitle": "通勤也能稳稳带走",
            "render_strategy": "text_to_video",
        }
    ]
    script_plan = {
        "hook": "出门前总担心水杯乱晃？",
        "body": ["侧袋收纳更稳", "大容量减少反复接水"],
        "cta": "适合每天通勤带水。",
    }

    monkeypatch.setattr(workflow, "_task_output_dir", lambda task_data, task_id: str(tmp_path))
    monkeypatch.setattr(
        workflow,
        "process_assets",
        lambda task_data: {
            "semantic_summary": "一张水杯图",
            "assets": [],
            "asset_profiles": [],
            "product_identity_card": {"identity_confidence": "high", "product_type": "水杯"},
        },
    )
    monkeypatch.setattr(workflow, "structurize_user_requirements", lambda task_data: {"input_confidence": "high"})
    monkeypatch.setattr(workflow, "build_asset_capability_plan", lambda asset_analysis, product_context: {})
    monkeypatch.setattr(workflow, "plan_storyboard_from_template", lambda product_context, asset_analysis: (storyboard, script_plan))
    monkeypatch.setattr(workflow, "review_storyboard_shootability", lambda *args, **kwargs: {"passed": True, "issues": []})
    monkeypatch.setattr(workflow, "_enforce_storyboard_continuity_groups", lambda value: value)
    monkeypatch.setattr(workflow, "_rule_based_narrative_review", lambda *args, **kwargs: {"passed": True, "issues": []})
    monkeypatch.setattr(workflow, "match_assets_to_storyboard", lambda storyboard, asset_analysis: [])
    monkeypatch.setattr(
        workflow,
        "complete_asset_gaps",
        lambda storyboard, asset_matching, asset_analysis, product_identity_card: {"asset_matching": asset_matching},
    )
    monkeypatch.setattr(
        workflow,
        "build_creation_plan",
        lambda product_context, storyboard, asset_matching: {"shots": storyboard, "total_duration_seconds": 4},
    )
    monkeypatch.setattr(workflow, "_save_workflow_artifacts", lambda task_id, output_dir, artifacts: str(tmp_path / "artifacts"))

    def fake_render(*args, **kwargs):
        nonlocal render_called
        render_called = True
        return {"success": True}

    monkeypatch.setattr(workflow, "render_seedance_video", fake_render)

    result = workflow.run_video_generation_workflow(
        {
            "task_id": "task-script-review",
            "title": "通勤水杯",
            "selling_points": ["稳固收纳", "大容量"],
            "target_platform": "tiktok",
            "duration_seconds": 15,
            "style": "lifestyle",
            "uploaded_assets": [],
        },
        stop_after_plan_review=True,
    )

    assert result["workflow_status"] == "needs_review"
    assert result["workflow_stage"] == "script_review"
    assert result["script_plan"]["hook"] == script_plan["hook"]
    assert result["script_plan"]["body"] == script_plan["body"]
    assert result["script_plan"]["cta"] == script_plan["cta"]
    assert result["storyboard"] == storyboard
    assert result["readable_script"]["hook"] == script_plan["hook"]
    assert result["script_plan"]["rich_story_text"]
    assert result["render_result"] == {}
    assert render_called is False


def test_approved_script_review_continues_with_edited_storyboard(monkeypatch, tmp_path):
    render_calls = []
    edited_storyboard = [
        {
            "shot_index": 1,
            "duration_seconds": 4,
            "narrative_role": "hook",
            "scene_goal": "展示收纳结果",
            "initial_state": "杯子在背包侧袋中",
            "action": "人物直接背包走过地铁口",
            "final_state": "杯子没有晃出",
            "subtitle": "通勤路上不怕晃",
            "render_strategy": "text_to_video",
        }
    ]
    approved_result = {
        "workflow_status": "needs_review",
        "workflow_stage": "script_review",
        "workflow_steps": [],
        "asset_analysis": {"assets": [], "product_identity_card": {"product_type": "水杯"}},
        "product_context": {"product_title": "通勤水杯", "product_type": "水杯", "selling_points": ["稳固"]},
        "script_plan": {"hook": "编辑后的开场", "body": ["编辑后的卖点"], "cta": "现在入手。"},
        "script_review": {"passed": True, "issues": []},
        "storyboard": edited_storyboard,
        "storyboard_review": {"passed": True, "issues": []},
        "review_attempts": 1,
        "asset_matching": [],
        "asset_gap_completion": {"asset_matching": []},
        "creation_plan": {"shots": edited_storyboard, "total_duration_seconds": 4},
        "narrative_review": {"passed": True, "issues": []},
        "narrative_review_attempts": [],
        "shootability_review": {"passed": True, "issues": []},
    }

    monkeypatch.setattr(workflow, "_task_output_dir", lambda task_data, task_id: str(tmp_path))
    monkeypatch.setattr(workflow, "build_creation_plan", lambda product_context, storyboard, asset_matching: {"shots": storyboard, "total_duration_seconds": 4})
    monkeypatch.setattr(workflow, "run_final_check", lambda *args, **kwargs: {"passed": True, "issues": []})
    monkeypatch.setattr(workflow, "_render_ab_variants", lambda **kwargs: {})
    monkeypatch.setattr(workflow, "review_rendered_video_content", lambda *args, **kwargs: {"passed": True, "issues": [], "shot_reviews": []})
    monkeypatch.setattr(workflow, "_save_workflow_artifacts", lambda task_id, output_dir, artifacts: str(tmp_path / "artifacts"))

    def fake_render_seedance_video(task_id, creation_plan, output_dir):
        render_calls.append(creation_plan)
        return {
            "success": True,
            "render_mode": "seedance",
            "video_path": str(tmp_path / "seedance_final.mp4"),
            "video_url": "/uploads/task-approved/seedance_final.mp4",
            "shot_results": [],
        }

    monkeypatch.setattr(workflow, "render_seedance_video", fake_render_seedance_video)

    result = workflow.continue_video_generation_workflow(
        {
            "task_id": "task-approved",
            "workflow_result": approved_result,
            "title": "通勤水杯",
            "selling_points": ["稳固"],
            "target_platform": "tiktok",
            "duration_seconds": 15,
            "uploaded_assets": [],
        }
    )

    assert result["workflow_status"] == "completed"
    assert result["workflow_stage"] == "draft_ready"
    assert result["workflow_progress"] == 100
    assert result["storyboard"][0]["subtitle"] == "通勤路上不怕晃"
    assert render_calls[0]["shots"][0]["subtitle"] == "通勤路上不怕晃"


def test_approved_script_review_renders_b_variant_even_with_legacy_selected_variant(monkeypatch, tmp_path):
    render_calls = []
    ab_called = False
    storyboard = [
        {
            "shot_index": 1,
            "duration_seconds": 5,
            "scene_goal": "展示玄关拿起水杯",
            "action": "手把水杯放进背包侧袋",
            "subtitle": "出门顺手带走",
            "render_strategy": "image_to_video",
        }
    ]
    approved_result = {
        "workflow_status": "needs_review",
        "workflow_stage": "script_review",
        "workflow_steps": [],
        "asset_analysis": {"assets": [], "product_identity_card": {"product_type": "水杯"}},
        "product_context": {"product_title": "通勤水杯", "product_type": "水杯", "selling_points": ["便携"]},
        "script_plan": {
            "hook": "B 开场",
            "body": ["B 卖点"],
            "cta": "B 结尾",
            "selected_review_variant": "B_ideal_commerce_scene",
        },
        "script_review": {"passed": True, "issues": []},
        "storyboard": storyboard,
        "storyboard_review": {"passed": True, "issues": []},
        "review_attempts": 1,
        "asset_matching": [],
        "asset_gap_completion": {"asset_matching": []},
        "creation_plan": {"shots": storyboard, "total_duration_seconds": 5},
        "narrative_review": {"passed": True, "issues": []},
        "narrative_review_attempts": [],
        "shootability_review": {"passed": True, "issues": []},
    }

    monkeypatch.setattr(workflow, "_task_output_dir", lambda task_data, task_id: str(tmp_path))
    monkeypatch.setattr(workflow, "build_creation_plan", lambda product_context, storyboard, asset_matching: {"shots": storyboard, "total_duration_seconds": 5})
    monkeypatch.setattr(workflow, "run_final_check", lambda *args, **kwargs: {"passed": True, "issues": []})
    monkeypatch.setattr(workflow, "review_rendered_video_content", lambda *args, **kwargs: {"passed": True, "issues": [], "shot_reviews": []})
    monkeypatch.setattr(workflow, "_save_workflow_artifacts", lambda task_id, output_dir, artifacts: str(tmp_path / "artifacts"))

    def fake_render_seedance_video(task_id, creation_plan, output_dir):
        render_calls.append(creation_plan)
        return {
            "success": True,
            "render_mode": "seedance",
            "video_path": str(tmp_path / "seedance_final.mp4"),
            "video_url": "/uploads/task-approved-b/seedance_final.mp4",
            "shot_results": [],
        }

    def fake_render_ab_variants(**kwargs):
        nonlocal ab_called
        ab_called = True
        return {"B_ideal_commerce_scene": {"success": True}}

    monkeypatch.setattr(workflow, "render_seedance_video", fake_render_seedance_video)
    monkeypatch.setattr(workflow, "_render_ab_variants", fake_render_ab_variants)

    result = workflow.continue_video_generation_workflow(
        {
            "task_id": "task-approved-b",
            "workflow_result": approved_result,
            "title": "通勤水杯",
            "selling_points": ["便携"],
            "target_platform": "tiktok",
            "duration_seconds": 15,
            "uploaded_assets": [],
        }
    )

    assert result["workflow_status"] == "completed"
    assert len(render_calls) == 1
    assert result["ab_variants"] == {"B_ideal_commerce_scene": {"success": True}}
    assert ab_called is True


def test_script_review_panel_renders_edit_approve_and_regenerate_controls():
    html = demo_app._render_script_review_panel(
        {
            "task_id": "task-ui-review",
            "workflow_stage": "script_review",
            "workflow_result": {
                "script_review_variants": {
                    "A_conservative_fidelity": {
                        "script_plan": {
                            "hook": "别再让水杯占满桌面",
                            "body": ["放进包侧袋", "路上稳定"],
                            "cta": "适合每天通勤。",
                        },
                        "storyboard": [
                            {
                                "shot_index": 1,
                                "duration_seconds": 4,
                                "scene_goal": "展示通勤场景",
                                "action": "用户背包走过地铁口",
                                "subtitle": "稳稳带走",
                            }
                        ],
                    },
                    "B_ideal_commerce_scene": {
                        "script_plan": {
                            "hook": "早高峰也能顺手带走",
                            "body": ["从玄关拿起", "放进背包侧袋"],
                            "cta": "每天出门都省心。",
                        },
                        "storyboard": [
                            {
                                "shot_index": 1,
                                "duration_seconds": 5,
                                "scene_goal": "用玄关场景展示便携",
                                "visual_description": "清晨玄关桌上放着同一个水杯，手从右侧进入，把水杯顺滑拿起后放进背包侧袋。",
                                "subtitle": "出门顺手带走",
                            }
                        ],
                    },
                },
            },
        }
    )

    assert "别再让水杯占满桌面" in html
    assert "早高峰也能顺手带走" in html
    assert "方案 A：稳妥保真版" in html
    assert "方案 B：场景带货版" in html
    assert "分镜时间轴" in html
    assert "总时长约 5 秒" in html
    assert "剧本已生成，等待你确认" in html
    assert 'form="script-review-form-task-ui-review"' in html
    assert 'action="/tasks/task-ui-review/script-review/approve"' in html
    assert 'name="selected_variant"' not in html
    assert 'type="radio"' not in html
    assert 'name="variant_A_conservative_fidelity__script_hook"' in html
    assert 'name="variant_A_conservative_fidelity__shot_0_action"' in html
    assert 'name="variant_B_ideal_commerce_scene__script_hook"' in html
    assert 'name="variant_B_ideal_commerce_scene__shot_0_action"' in html
    assert "清晨玄关桌上放着同一个水杯" in html
    assert 'formaction="/tasks/task-ui-review/script-review/regenerate"' in html
    assert "重新生成剧本" in html


def test_script_review_form_parses_both_editable_variants_without_selected_review_variant():
    script_plan, storyboard, edited_variants = demo_app._parse_script_review_submission(
        {
            "variant_A_conservative_fidelity__script_synopsis": "A 总剧本",
            "variant_A_conservative_fidelity__script_hook": "A 开场",
            "variant_A_conservative_fidelity__script_body": "A 卖点一\nA 卖点二",
            "variant_A_conservative_fidelity__script_cta": "A 结尾",
            "variant_A_conservative_fidelity__shot_count": "1",
            "variant_A_conservative_fidelity__shot_0_duration": "3",
            "variant_A_conservative_fidelity__shot_0_scene_goal": "A 目标",
            "variant_A_conservative_fidelity__shot_0_action": "A 画面",
            "variant_A_conservative_fidelity__shot_0_subtitle": "A 字幕",
            "variant_B_ideal_commerce_scene__script_synopsis": "B 总剧本",
            "variant_B_ideal_commerce_scene__script_hook": "B 开场",
            "variant_B_ideal_commerce_scene__script_body": "B 卖点一\nB 卖点二",
            "variant_B_ideal_commerce_scene__script_cta": "B 结尾",
            "variant_B_ideal_commerce_scene__shot_count": "1",
            "variant_B_ideal_commerce_scene__shot_0_duration": "5",
            "variant_B_ideal_commerce_scene__shot_0_scene_goal": "B 目标",
            "variant_B_ideal_commerce_scene__shot_0_action": "B 画面",
            "variant_B_ideal_commerce_scene__shot_0_subtitle": "B 字幕",
        },
        {
            "script_review_variants": {
                "A_conservative_fidelity": {
                    "script_plan": {"hook": "旧 A"},
                    "storyboard": [{"shot_index": 1, "duration_seconds": 3, "subtitle": "旧 A"}],
                },
                "B_ideal_commerce_scene": {
                    "script_plan": {"hook": "旧 B"},
                    "storyboard": [{"shot_index": 1, "duration_seconds": 5, "subtitle": "旧 B"}],
                },
            }
        },
    )

    assert set(edited_variants) == {"A_conservative_fidelity", "B_ideal_commerce_scene"}
    assert "selected_review_variant" not in script_plan
    assert script_plan["hook"] == "A 开场"
    assert script_plan["body"] == ["A 卖点一", "A 卖点二"]
    assert storyboard == [
        {
            "shot_index": 1,
            "duration_seconds": 3,
            "subtitle": "A 字幕",
            "scene_goal": "A 目标",
            "action": "A 画面",
            "review_variant_id": "A_conservative_fidelity",
        }
    ]
    assert edited_variants["A_conservative_fidelity"]["script_plan"]["hook"] == "A 开场"
    assert edited_variants["B_ideal_commerce_scene"]["script_plan"]["hook"] == "B 开场"
    assert "selected_review_variant" not in edited_variants["A_conservative_fidelity"]["script_plan"]
    assert "selected_review_variant" not in edited_variants["B_ideal_commerce_scene"]["script_plan"]
    assert edited_variants["B_ideal_commerce_scene"]["storyboard"] == [
        {
            "shot_index": 1,
            "duration_seconds": 5,
            "subtitle": "B 字幕",
            "scene_goal": "B 目标",
            "action": "B 画面",
            "review_variant_id": "B_ideal_commerce_scene",
        }
    ]


def test_render_ab_variant_uses_user_edited_b_storyboard(monkeypatch, tmp_path):
    edited_b_storyboard = [
        {
            "shot_index": 1,
            "duration_seconds": 5,
            "scene_goal": "B 编辑后的场景目标",
            "action": "B 编辑后的画面描述",
            "subtitle": "B 编辑后的字幕",
            "render_strategy": "text_to_video",
        }
    ]
    render_calls = []

    monkeypatch.delenv("AIGC_DISABLE_AB_VARIANTS", raising=False)
    monkeypatch.setattr(workflow, "_task_output_dir", lambda task_data, task_id: str(tmp_path))
    monkeypatch.setattr(
        workflow,
        "_plan_ideal_commerce_scene_storyboard",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("不应重新规划 B 方案")),
    )
    monkeypatch.setattr(workflow, "match_assets_to_storyboard", lambda storyboard, asset_analysis: [])
    monkeypatch.setattr(
        workflow,
        "complete_asset_gaps",
        lambda storyboard, asset_matching, asset_analysis, product_identity_card: {"asset_matching": asset_matching},
    )
    monkeypatch.setattr(
        workflow,
        "build_creation_plan",
        lambda product_context, storyboard, asset_matching: {"shots": storyboard, "total_duration_seconds": 5},
    )

    def fake_render_seedance_video(task_id, creation_plan, output_dir):
        render_calls.append({"creation_plan": creation_plan, "output_dir": output_dir})
        return {
            "success": True,
            "video_path": str(tmp_path / "variants" / "B_ideal_commerce_scene" / "seedance_final.mp4"),
            "video_url": "/uploads/task-b/variants/B_ideal_commerce_scene/seedance_final.mp4",
        }

    monkeypatch.setattr(workflow, "render_seedance_video", fake_render_seedance_video)

    result = workflow._render_ab_variants(
        task_id="task-b",
        task_data={"task_id": "task-b"},
        product_context={"product_identity_card": {"product_type": "水杯"}},
        asset_analysis={"assets": []},
        script_review_variants={
            "B_ideal_commerce_scene": {
                "script_plan": {"hook": "B 编辑后的开场"},
                "storyboard": edited_b_storyboard,
            }
        },
    )

    b_variant = result["B_ideal_commerce_scene"]
    assert b_variant["storyboard"] == edited_b_storyboard
    assert b_variant["script_plan"]["hook"] == "B 编辑后的开场"
    assert render_calls[0]["creation_plan"]["shots"] == edited_b_storyboard
    assert Path(render_calls[0]["output_dir"]).parts[-2:] == ("variants", "B_ideal_commerce_scene")


def test_script_regeneration_feedback_includes_current_edited_shots():
    script_plan, storyboard = demo_app._parse_script_review_form(
        {
            "script_synopsis": "总剧本",
            "script_hook": "开场",
            "script_body": "卖点",
            "script_cta": "结尾",
            "shot_count": "1",
            "shot_0_duration": "4",
            "shot_0_scene_goal": "展示通勤痛点",
            "shot_0_action": "改成放进包侧袋后走过地铁口，不要只摸杯子",
            "shot_0_subtitle": "通勤稳稳带走",
        },
        {
            "script_plan": {"hook": "旧开场", "body": [], "cta": ""},
            "storyboard": [{"shot_index": 1, "duration_seconds": 4}],
        },
    )
    feedback = demo_app._compose_regeneration_feedback(
        "第二镜头更有剧情",
        script_plan,
        storyboard,
    )

    assert "第二镜头更有剧情" in feedback
    assert "改成放进包侧袋后走过地铁口" in feedback
    assert "不要只摸杯子" in feedback


def test_result_page_renders_a_b_video_comparison():
    html = demo_app._render_workflow_result(
        {
            "workflow_status": "completed",
            "workflow_stage": "draft_ready",
            "script_plan": {"hook": "开场", "body": ["卖点"], "cta": "行动"},
            "storyboard": [],
            "creation_plan": {},
            "render_result": {
                "success": True,
                "render_mode": "seedance",
                "video_url": "/uploads/task-ab/seedance_final.mp4",
                "video_path": ".uploads/task-ab/seedance_final.mp4",
            },
            "ab_variants": {
                "B_ideal_commerce_scene": {
                    "success": True,
                    "strategy": "B_ideal_commerce_scene",
                    "render_result": {
                        "success": True,
                        "video_url": "/uploads/task-ab/variants/B_ideal_commerce_scene/seedance_final.mp4",
                    },
                    "review_notes": ["更偏场景化卖点表达"],
                }
            },
            "final_check": {"passed": True, "issues": []},
            "content_review": {"passed": True},
            "trace_summary": {},
        }
    )

    assert "方案 A" in html
    assert "方案 B" in html
    assert "/uploads/task-ab/seedance_final.mp4" in html
    assert "/uploads/task-ab/variants/B_ideal_commerce_scene/seedance_final.mp4" in html
    assert "下载视频" in html
    assert "新窗口预览" in html
    assert 'download="product_video_a.mp4"' in html


def test_generation_progress_card_shows_stage_checklist_and_resume_hint():
    html = demo_app._render_generation_progress_card(
        {
            "workflow_stage": "render_video",
            "workflow_message": "正在调用视频模型生成分镜视频。",
            "workflow_progress": 78,
            "workflow_events": [
                {"stage": "asset_analysis", "progress": 20, "message": "素材理解完成", "created_at": "now"},
                {"stage": "render_video", "progress": 78, "message": "正在生成视频", "created_at": "now"},
            ],
        }
    )

    assert "可关闭页面后继续查看" in html
    assert "理解素材" in html
    assert "生成视频" in html
    assert "检查成片" in html
    assert "最近进度" in html


def test_detail_page_renders_stage_rail_and_copy_task_link_action():
    html = demo_app._render_page(
        success_task={
            "task_id": "task-progress-rail",
            "status": "processing",
            "workflow_stage": "render_video",
            "workflow_message": "正在生成视频",
            "workflow_progress": 78,
            "workflow_events": [
                {"stage": "asset_analysis", "progress": 20, "message": "素材理解完成", "created_at": "now"},
                {"stage": "render_video", "progress": 78, "message": "正在生成视频", "created_at": "now"},
            ],
            "workflow_result": {},
            "uploaded_assets": [],
            "selling_points": ["便携"],
        },
        page_mode="detail",
    )

    assert 'id="stage-rail"' in html
    assert "复制任务链接" in html
    assert "导出任务报告" in html
    assert "/tasks/task-progress-rail/report.json" in html
    assert "copyCurrentTaskLink" in html
    assert "视频渲染" in html


def test_health_endpoint_reports_runtime_state_without_secrets(monkeypatch):
    repository = InMemoryTaskRepository()
    create_video_task(
        CreateVideoTaskCommand(
            title="水杯",
            selling_points=["大容量"],
            target_platform="tiktok",
            duration_seconds=15,
            style="product_showcase",
        ),
        repository,
    )
    monkeypatch.setattr(demo_app, "repository", repository)
    monkeypatch.setenv("ARK_API_KEY", "secret-key")
    monkeypatch.setenv("ARK_TEXT_ENDPOINT_ID", "text-endpoint")
    monkeypatch.setenv("ARK_VIDEO_ENDPOINT_ID", "video-endpoint")
    monkeypatch.setenv("AIGC_DISABLE_LLM", "1")

    data = demo_app.api_health()

    assert data["status"] == "ok"
    assert data["server_pid"] == demo_app.SERVER_PID
    assert data["run_instance_id"] == demo_app.RUN_INSTANCE_ID
    assert data["port"] == 8010
    assert data["task_count"] == 1
    assert data["disable_llm"] is True
    assert data["ark_text_configured"] is True
    assert data["ark_video_configured"] is True
    assert "secret-key" not in json.dumps(data)
    assert "text-endpoint" not in json.dumps(data)


def test_listener_pid_parser_matches_ss_loopback_port(monkeypatch):
    class Completed:
        stdout = (
            "State Recv-Q Send-Q Local Address:Port Peer Address:Port Process\n"
            'LISTEN 0 2048 127.0.0.1:8010 0.0.0.0:* users:(("python",pid=12345,fd=6))\n'
        )

    monkeypatch.setattr(demo_app.subprocess, "run", lambda *args, **kwargs: Completed())

    assert demo_app._listener_pids_on_port(8010) == {12345}


def test_task_report_json_exports_reviewable_summary(monkeypatch):
    repository = InMemoryTaskRepository()
    task = create_video_task(
        CreateVideoTaskCommand(
            title="笔记本",
            selling_points=["轻薄"],
            target_platform="tiktok",
            duration_seconds=15,
            style="product_showcase",
        ),
        repository,
    )
    task.workflow_result = {
        "workflow_status": "completed",
        "workflow_stage": "draft_ready",
        "artifacts_dir": ".uploads/task-report/artifacts",
        "script_plan": {"hook": "轻薄随行"},
        "render_result": {
            "success": True,
            "video_url": "/uploads/task-report/seedance_final.mp4",
            "video_path": ".uploads/task-report/seedance_final.mp4",
            "render_mode": "seedance",
            "elapsed_seconds": 12.5,
        },
        "ab_variants": {
            "B_ideal_commerce_scene": {
                "success": True,
                "render_result": {
                    "success": True,
                    "video_url": "/uploads/task-report/variants/B_ideal_commerce_scene/seedance_final.mp4",
                    "video_path": ".uploads/task-report/variants/B_ideal_commerce_scene/seedance_final.mp4",
                },
            }
        },
    }
    repository.update(task)
    monkeypatch.setattr(demo_app, "repository", repository)

    report = demo_app.task_report_json(task.task_id)

    assert report["task"]["task_id"] == task.task_id
    assert "workflow_result" not in report["task"]
    assert report["workflow_result"]["script_plan"]["hook"] == "轻薄随行"
    assert report["artifact_dir"] == ".uploads/task-report/artifacts"
    assert [item["label"] for item in report["video_urls"]] == ["A_default", "B_ideal_commerce_scene"]
    assert report["video_urls"][0]["video_url"] == "/uploads/task-report/seedance_final.mp4"


def test_variant_video_url_prefers_video_path_when_video_url_points_to_default_a(tmp_path):
    variant_path = tmp_path / ".uploads" / "task-ab" / "variants" / "B_ideal_commerce_scene" / "seedance_final.mp4"

    assert demo_app._video_url_from_result({
        "video_url": "/uploads/task-ab/seedance_final.mp4",
        "video_path": str(variant_path),
    }) == "/uploads/task-ab/variants/B_ideal_commerce_scene/seedance_final.mp4"


def test_seedance_public_video_url_uses_nested_variant_output_path(tmp_path):
    video_path = tmp_path / ".uploads" / "task-ab" / "variants" / "B_ideal_commerce_scene" / "seedance_final.mp4"

    assert _public_upload_url_for_video(video_path, "task-ab") == (
        "/uploads/task-ab/variants/B_ideal_commerce_scene/seedance_final.mp4"
    )


def test_detail_page_uses_api_polling_for_processing_tasks():
    html = demo_app._build_full_page("<main></main>", json.dumps("task-poll"), True)

    assert "/api/tasks/" in html
    assert "fetch(" in html


def test_create_page_renders_commerce_style_templates():
    html = demo_app._render_page()

    assert "选择带货风格模板" in html
    assert "科技感未来风" in html
    assert "高质感奢华风" in html
    assert "清新自然生活风" in html
    assert "卡通趣味可爱风" in html
    assert "快节奏种草剪辑" in html
    assert 'name="style_template_id"' in html
    assert "selectTrendTemplate" in html
    assert "custom-style-prompt" in html
    assert "/uploads/style_templates/tech_future_showcase.mp4" in html
    assert "controls autoplay muted loop playsinline" in html
    assert "样例视频" in html
    assert "data-prompt" not in html


def test_selected_style_template_merges_into_custom_style_prompt():
    prompt = demo_app._compose_template_style_prompt(
        "tech_future_showcase",
        "用户希望节奏更慢，字幕更少。",
    )

    assert "【模板：科技感未来风】" in prompt
    assert "用户补充：用户希望节奏更慢，字幕更少。" in prompt
    assert "看看这个好物" not in prompt


def test_style_template_overrides_base_style_to_avoid_conflicting_prompts():
    assert demo_app._effective_style_value("tech_future_showcase", "premium") == "product_showcase"
    prompt = demo_app._compose_template_style_prompt("tech_future_showcase", "节奏稍慢")

    style_bible = workflow._build_visual_style_bible(
        {
            "style": demo_app._effective_style_value("tech_future_showcase", "premium"),
            "custom_style_prompt": prompt,
        }
    )

    assert style_bible["style_summary"] == "科技感未来风；用户补充：节奏稍慢"
    assert "premium" not in style_bible["user_style"]
    assert "不要写死某个使用场景" not in style_bible["lighting"]


def test_style_template_selection_can_be_restored_on_form_error():
    html = demo_app._render_page(form_values={"style_template_id": "premium_luxury_texture"})

    assert 'id="f-style-template" value="premium_luxury_texture"' in html
    assert 'class="trend-template-card selected"' in html


def test_script_review_panel_shows_full_story_synopsis_before_shots():
    html = demo_app._render_script_review_panel(
        {
            "task_id": "task-rich-script",
            "workflow_stage": "script_review",
            "workflow_result": {
                "script_plan": {
                    "rich_story_text": "这条视频讲一个上班族早晨出门，把轻薄笔记本放进通勤包，到地铁口仍然轻松移动的故事。",
                    "hook": "早高峰也不用背得很累",
                    "body": ["轻薄随行", "性能在线"],
                    "cta": "适合每天通勤。",
                },
                "storyboard": [
                    {
                        "shot_index": 1,
                        "duration_seconds": 4,
                        "scene_goal": "早晨出门前的通勤痛点",
                        "action": "用户把笔记本放入通勤包",
                        "subtitle": "轻薄随行",
                    }
                ],
            },
        }
    )

    assert "总剧本" in html
    assert "这条视频讲一个上班族早晨出门" in html
    assert html.index("总剧本") < html.index("分镜 1")


def test_script_review_detail_page_is_focused_and_does_not_stack_workflow_output():
    html = demo_app._render_page(
        success_task={
            "task_id": "task-review-page",
            "status": "needs_review",
            "workflow_stage": "script_review",
            "workflow_message": "剧本和分镜已生成",
            "workflow_progress": 72,
            "workflow_result": {
                "script_plan": {
                    "rich_story_text": "一个完整的通勤卖点故事。",
                    "hook": "早高峰也不用背得很累",
                    "body": ["轻薄随行"],
                    "cta": "适合每天通勤。",
                },
                "storyboard": [
                    {
                        "shot_index": 1,
                        "duration_seconds": 4,
                        "scene_goal": "通勤痛点",
                        "action": "用户背包走过地铁口",
                        "subtitle": "轻薄随行",
                    }
                ],
            },
            "uploaded_assets": [],
            "selling_points": ["轻薄随行"],
        },
        page_mode="detail",
    )

    assert "确认剧本" in html
    assert 'id="workflow-output"' not in html
    assert "AI 导演决策" not in html


def test_result_page_shows_strategy_summary_when_director_decision_is_empty():
    html = demo_app._render_workflow_result(
        {
            "asset_analysis": {},
            "product_context": {
                "product_title": "雷蛇笔记本",
                "creative_goal": "展示轻薄高性能",
                "visual_style_bible": {"style_summary": "科技感未来风"},
            },
            "script_plan": {"hook": "轻薄也能高性能", "body": ["移动办公"], "cta": "适合日常随身。"},
            "storyboard": [],
            "creation_plan": {"render_mode": "seedance_auto_with_local_fallback"},
            "render_result": {},
            "director_decision": {},
            "final_check": {},
        }
    )

    assert "创作策略摘要" in html
    assert "AI 导演决策" not in html
    assert "科技感未来风" in html


def test_task_polling_script_uses_valid_json_task_id_assignment():
    html = demo_app._build_full_page("<main></main>", json.dumps("task-poll"), True)

    assert 'const TASK_ID = "task-poll";' in html
    assert 'const TASK_ID = ""task-poll"";' not in html


def test_create_page_submit_button_says_generate_script_first():
    html = demo_app._render_page()

    assert "生成剧本" in html
    assert "开始生成视频" not in html


def test_create_page_accepts_video_files_and_video_links():
    html = demo_app._render_page()

    assert 'accept="image/*,video/*"' in html
    assert 'name="video_urls"' in html
    assert "视频链接" in html


def test_uploaded_asset_serializes_video_source_url_and_asset_type():
    repository = InMemoryTaskRepository()
    task = create_video_task(
        CreateVideoTaskCommand(
            title="通勤水杯",
            selling_points=["便携"],
            target_platform="tiktok",
            duration_seconds=15,
            style="product_showcase",
            uploaded_assets=[
                UploadedAsset(
                    filename="reference.mp4",
                    content_type="video/mp4",
                    asset_type="video",
                    source_url="https://example.com/reference.mp4",
                )
            ],
        ),
        repository,
    )

    asset = task.to_dict()["uploaded_assets"][0]
    assert asset["asset_type"] == "video"
    assert asset["source_url"] == "https://example.com/reference.mp4"


def test_video_link_assets_parse_http_urls_as_external_video_assets():
    assets = demo_app._video_link_assets(
        "https://cdn.example.com/demo.mp4\n"
        "not-a-url\n"
        "https://example.com/path/watch?id=123"
    )

    assert [asset.source_url for asset in assets] == [
        "https://cdn.example.com/demo.mp4",
        "https://example.com/path/watch?id=123",
    ]
    assert all(asset.asset_type == "video" for asset in assets)
    assert all(asset.content_type == "video/external" for asset in assets)


def test_normalize_asset_keeps_external_video_source_url_supported():
    asset = workflow._normalize_asset(
        {
            "filename": "external-video",
            "content_type": "video/external",
            "asset_type": "video",
            "source_url": "https://example.com/product-demo.mp4",
            "public_url": "https://example.com/product-demo.mp4",
        }
    )

    assert asset["asset_type"] == "video"
    assert asset["is_supported"] is True
    assert asset["source_url"] == "https://example.com/product-demo.mp4"
    assert asset["suggested_role"] == "商品视频片段候选"


def test_template_script_synopsis_uses_product_context_not_only_stage_names():
    script = workflow._build_template_script_plan_stub(
        [
            {
                "shot_index": 1,
                "duration_seconds": 4,
                "narrative_role": "hook",
                "scene_goal": "使用情境",
                "action": "用户把水杯放进背包侧袋并走向地铁口",
                "subtitle": "通勤稳稳带走",
            }
        ],
        product_type="水杯",
        source="template_path_b_no_anchor_director",
        product_context={
            "product_title": "通勤水杯",
            "usage_scene": "早高峰通勤",
            "target_audience": "上班族",
            "selling_points": ["轻巧便携", "大容量"],
        },
    )

    assert "上班族" in script["rich_story_text"]
    assert "早高峰通勤" in script["rich_story_text"]
    assert "轻巧便携" in script["rich_story_text"]
    assert "核心卖点" not in script["rich_story_text"]


def test_template_script_synopsis_does_not_treat_camera_motion_as_story_action():
    script = workflow._build_template_script_plan_stub(
        [
            {
                "shot_index": 1,
                "duration_seconds": 4,
                "narrative_role": "hook",
                "scene_goal": "使用情境",
                "action": "定格",
                "subtitle": "通勤稳稳带走",
            }
        ],
        product_type="水杯",
        source="template_path_b_no_anchor_director",
        product_context={
            "product_title": "通勤水杯",
            "usage_scene": "早高峰通勤",
            "target_audience": "上班族",
            "selling_points": ["轻巧便携", "大容量"],
        },
    )

    assert "定格" not in script["rich_story_text"]
    assert "先建立在早高峰通勤的需求" in script["rich_story_text"]


def test_ideal_commerce_variant_uses_visual_proof_subtitles_not_raw_long_selling_points():
    storyboard = workflow._plan_ideal_commerce_scene_storyboard(
        {
            "product_identity_card": {
                "product_type": "笔记本电脑",
                "appearance_summary": "黑色闭合笔记本，A 面有绿色 logo",
                "visible_marks": ["绿色 logo", "RAZER 字样"],
                "key_components": ["闭合机身", "A 面 logo"],
            },
            "selling_points": ["高性能处理器，流畅不卡顿,精致做工，品质感拉满,超薄轻巧，随时携带"],
            "usage_scene": "通勤办公",
            "target_audience": "上班族",
        },
        {
            "asset_profiles": [
                {
                    "asset_id": "laptop_anchor",
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                }
            ],
            "assets": [
                {
                    "asset_id": "laptop_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/laptop_anchor.png",
                    "is_supported": True,
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                }
            ],
        },
    )

    subtitles = [shot["subtitle"] for shot in storyboard]
    prompts = [shot["video_prompt"] for shot in storyboard]
    strategies = [shot.get("expression_strategy") for shot in storyboard]
    forbidden_public_lines = {"看看这个好物", "点击了解更多", "点击查看详情", "真实外观确认"}
    fixed_laptop_lines = ["轻薄好收纳", "轻薄好收纳", "移动办公更省心"]

    assert len(storyboard) == 3
    assert subtitles != fixed_laptop_lines
    assert not forbidden_public_lines & set(subtitles)
    assert all(strategies)
    assert set(strategies) <= {
        "direct_benefit_proof",
        "usage_result_demo",
        "premium_texture_reveal",
        "scene_fit_showcase",
        "feature_operation_demo",
        "problem_solution_pair",
        "aspirational_lifestyle_result",
        "identity_material_confirm",
    }
    assert "画面里看出卖点" in prompts[1]
    assert "画面证明" in prompts[2]
    assert all(strategy not in " ".join(prompts) for strategy in strategies)
    assert all("高性能处理器，流畅不卡顿" not in subtitle for subtitle in subtitles)
    assert "不要新增非商品自带文字" in prompts[0]
    assert "不要生成画面内文字" not in prompts[0]
    assert "随机文字" in prompts[2]


def test_ideal_commerce_water_cup_variant_keeps_result_shot_required_and_not_forbidden():
    storyboard = workflow._plan_ideal_commerce_scene_storyboard(
        {
            "product_identity_card": {
                "product_type": "塑料吸管水杯",
                "appearance_summary": "透明杯身，彩色杯盖，带吸管和黄色挂饰",
                "visible_marks": ["CHAKO LAB"],
                "key_components": ["吸管", "翻盖杯盖", "挂饰"],
                "motion_affordance": {
                    "can_be_handheld": True,
                    "allowed_actions": ["手持", "放在桌面"],
                    "forbidden_actions": [],
                },
            },
            "selling_points": ["颜值在线，多色可选", "容量大", "冷热都能装"],
            "usage_scene": "户外",
        },
        {
            "asset_profiles": [
                {
                    "asset_id": "cup_anchor",
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                }
            ],
            "assets": [
                {
                    "asset_id": "cup_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/cup_anchor.png",
                    "is_supported": True,
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                }
            ],
        },
    )

    assert len(storyboard) == 3
    assert all(shot["required_for_variant"] is True for shot in storyboard)
    assert storyboard[1]["selected_prompt_skill"] == "commerce_scene.material_action_proof"
    assert storyboard[2]["selected_prompt_skill"] == "commerce_scene.new_scene_result"
    assert storyboard[2]["render_strategy"] == "text_to_video"
    assert storyboard[2]["product_presence"] != "forbidden"
    assert "触摸" not in storyboard[1]["action"]
    assert "手指" not in storyboard[1]["action"]
    assert any(word in storyboard[2]["visual_description"] for word in ["户外", "随身", "补水", "冰饮"])


def test_build_creation_plan_keeps_new_scene_result_product_optional():
    storyboard = [
        {
            "shot_index": 2,
            "duration_seconds": 5,
            "narrative_role": "commerce_result_scene",
            "render_strategy": "text_to_video",
            "product_presence": "optional",
            "identity_strictness": "medium",
            "scene_goal": "用户外结果状态证明水杯随身补水",
            "initial_state": "硬切后的新场景",
            "action": "人物只做轻微辅助动作",
            "final_state": "水杯仍清楚可见",
            "camera_motion": "定镜",
            "visual_description": "同一件塑料吸管水杯已经在户外随身包侧袋旁，透明杯身和冰饮清楚可见。",
            "subtitle": "出门随身补水",
            "voiceover": "出门随身补水",
            "asset_requirement": "不使用首帧素材，靠详细外观描述生成新场景结果镜。",
            "selected_prompt_skill": "commerce_scene.new_scene_result",
            "force_video_prompt": True,
            "video_prompt": "同一件塑料吸管水杯已经在户外随身包侧袋旁，透明杯身和冰饮清楚可见。",
            "required_for_variant": True,
            "forbidden_variation": ["第二个同类主商品", "错误 logo"],
            "review_focus": ["商品身份", "卖点是否由画面证明"],
        }
    ]

    plan = workflow.build_creation_plan(
        {"target_platform": "tiktok", "product_identity_card": {"product_type": "塑料吸管水杯"}},
        storyboard,
        [],
    )

    shot = plan["shots"][0]
    assert shot["selected_prompt_skill"] == "commerce_scene.new_scene_result"
    assert shot["render_strategy"] == "text_to_video"
    assert shot["product_presence"] == "optional"
    assert shot["required_for_variant"] is True
    assert "塑料吸管水杯" in shot["render_input"]["prompt"]


def test_ideal_commerce_variant_selects_generic_strategy_for_skincare_operation_demo():
    storyboard = workflow._plan_ideal_commerce_scene_storyboard(
        {
            "product_identity_card": {
                "product_type": "修护精华乳",
                "appearance_summary": "半透明磨砂瓶身，银色泵头，浅蓝标签",
                "visible_marks": ["浅蓝标签", "银色泵头"],
                "key_components": ["银色泵头", "磨砂瓶身"],
                "material_features": ["磨砂玻璃质感", "细腻乳霜质地"],
                "functional_features": ["按压泵头定量取用", "夜间修护保湿"],
                "motion_affordance": {
                    "can_be_handheld": True,
                    "allowed_actions": ["按压泵头少量取用"],
                    "forbidden_actions": [],
                },
            },
            "selling_points": ["磨砂玻璃质感", "按压泵头定量取用", "夜间修护保湿"],
            "usage_scene": "晚间梳妆台护肤",
            "target_audience": "干皮通勤人群",
        },
        {
            "asset_profiles": [
                {
                    "asset_id": "skincare_anchor",
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                }
            ],
            "assets": [
                {
                    "asset_id": "skincare_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/skincare_anchor.png",
                    "is_supported": True,
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                }
            ],
        },
    )

    subtitles = [shot["subtitle"] for shot in storyboard]
    prompts = [shot["video_prompt"] for shot in storyboard]
    strategies = [shot.get("expression_strategy") for shot in storyboard]
    forbidden_public_lines = {"看看这个好物", "点击了解更多", "点击查看详情", "真实外观确认"}

    assert len(storyboard) == 3
    assert set(strategies) == {"feature_operation_demo"}
    assert "按压泵头定量取用" in subtitles
    assert "夜间修护保湿" in subtitles
    assert not forbidden_public_lines & set(subtitles)
    assert "按压泵头定量取用" in prompts[1]
    assert "skill" not in prompts[1]
    assert "银色泵头" in prompts[1]
    assert "梳妆台" in prompts[2]
    assert "泛生活场景" in prompts[2]
    assert "看看这个好物" not in "".join(prompts + subtitles)
    assert "点击了解更多" not in "".join(prompts + subtitles)
    assert all(strategy not in " ".join(prompts) for strategy in strategies)


def test_ideal_commerce_variant_uses_llm_skill_expression_plan(monkeypatch):
    monkeypatch.setenv("AIGC_TEST_ALLOW_LLM", "1")

    def fake_call_text_llm(prompt_data, purpose, temperature=0.7):
        assert purpose == "commerce_expression_plan"
        assert "Commerce Expression Strategies" in prompt_data["strategy_reference"]
        return {
            "ok": True,
            "content": json.dumps(
                {
                    "expression_strategy": "premium_texture_reveal",
                    "primary_value": "金属质感",
                    "result_value": "桌面更有品质感",
                    "confirm_caption": "金属质感",
                    "action_caption": "边缘细节清楚",
                    "result_caption": "桌面更有品质感",
                    "source_place": "原素材的深色桌面",
                    "result_place": "极简办公桌",
                    "human": "只出现用户手部整理桌面",
                    "source_action": "手指先扶住商品边缘，再沿金属倒角短距离滑过，最后停在材质近景。",
                    "result_action": "手部只把旁边记事本摆正，商品保持居中清楚。",
                    "result_state": "商品位于极简办公桌中心，金属边缘反光和桌面道具形成品质感结果。",
                    "notes_for_review": "用质感表达，不套痛点解决。",
                },
                ensure_ascii=False,
            ),
        }

    monkeypatch.setattr(workflow, "_call_text_llm", fake_call_text_llm)

    storyboard = workflow._plan_ideal_commerce_scene_storyboard(
        {
            "product_identity_card": {
                "product_type": "无线充电底座",
                "appearance_summary": "深灰金属圆形底座，边缘倒角清楚",
                "material_features": ["金属倒角", "哑光表面"],
                "key_components": ["圆形底座"],
            },
            "selling_points": ["金属质感", "桌面更有品质感"],
            "usage_scene": "办公桌面",
        },
        {
            "asset_profiles": [
                {
                    "asset_id": "dock_anchor",
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                    "reasoning": "完整商品和桌面可见。",
                }
            ],
            "assets": [
                {
                    "asset_id": "dock_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/dock.png",
                    "is_supported": True,
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                }
            ],
        },
    )

    assert len(storyboard) == 3
    assert {shot["expression_strategy"] for shot in storyboard} == {"premium_texture_reveal"}
    assert {shot["expression_plan_source"] for shot in storyboard} == {"llm_skill_plan"}
    assert [shot["subtitle"] for shot in storyboard] == ["金属质感", "边缘细节清楚", "桌面更有品质感"]
    assert "原素材的深色桌面" in storyboard[0]["video_prompt"]
    assert "手指先扶住商品边缘" in storyboard[1]["video_prompt"]
    assert "极简办公桌" in storyboard[2]["video_prompt"]
    assert "premium_texture_reveal" not in " ".join(shot["video_prompt"] for shot in storyboard)


def test_ideal_commerce_variant_repairs_unprovable_llm_result_plan(monkeypatch):
    monkeypatch.setenv("AIGC_TEST_ALLOW_LLM", "1")

    def fake_call_text_llm(prompt_data, purpose, temperature=0.7):
        assert purpose == "commerce_expression_plan"
        return {
            "ok": True,
            "content": json.dumps(
                {
                    "expression_strategy": "direct_benefit_proof",
                    "primary_value": "高性能处理器",
                    "result_value": "长续航",
                    "confirm_caption": "高性能办公",
                    "action_caption": "处理更流畅",
                    "result_caption": "长续航",
                    "source_place": "素材桌面",
                    "result_place": "咖啡店桌面",
                    "human": "普通办公用户",
                    "source_action": "手指轻触触控板，屏幕保持办公软件运行。",
                    "result_action": "手将笔记本从包中取出，放在桌上。",
                    "result_state": "笔记本位于咖啡店桌面，旁边有背包和咖啡。",
                    "notes_for_review": "字段完整但结果动作无法证明续航。",
                },
                ensure_ascii=False,
            ),
        }

    monkeypatch.setattr(workflow, "_call_text_llm", fake_call_text_llm)

    storyboard = workflow._plan_ideal_commerce_scene_storyboard(
        {
            "product_identity_card": {
                "product_type": "笔记本电脑",
                "appearance_summary": "银色轻薄笔记本，屏幕打开，窄边框",
                "key_components": ["屏幕", "键盘", "触控板"],
                "functional_features": ["长续航", "移动办公"],
            },
            "selling_points": ["高性能处理器", "长续航"],
            "usage_scene": "咖啡店移动办公",
            "target_audience": "通勤办公人群",
        },
        {
            "asset_profiles": [{"asset_id": "laptop_anchor", "visual_role": "appearance_anchor", "quality_score": 90}],
            "assets": [
                {
                    "asset_id": "laptop_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/laptop_anchor.png",
                    "is_supported": True,
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                }
            ],
        },
    )

    result_shot = storyboard[2]
    evidence_text = result_shot["action"] + result_shot["visual_description"] + result_shot["video_prompt"]
    assert result_shot["expression_plan_source"] == "llm_skill_plan"
    assert result_shot["subtitle"] == "长续航"
    assert "不插电" in evidence_text
    assert "电源线" in evidence_text or "充电器" in evidence_text
    assert "从包中取出，放在桌上" not in result_shot["action"]


def test_ideal_commerce_variant_marks_fallback_when_llm_plan_fails(monkeypatch):
    monkeypatch.setenv("AIGC_TEST_ALLOW_LLM", "1")

    def fake_call_text_llm(prompt_data, purpose, temperature=0.7):
        return {"ok": False, "content": "", "error": "backend down"}

    monkeypatch.setattr(workflow, "_call_text_llm", fake_call_text_llm)

    storyboard = workflow._plan_ideal_commerce_scene_storyboard(
        {
            "product_identity_card": {
                "product_type": "收纳盒",
                "appearance_summary": "透明方形收纳盒，白色盖子",
                "key_components": ["盒盖", "透明盒身"],
            },
            "selling_points": ["桌面更整齐", "透明可视"],
            "usage_scene": "办公桌收纳",
        },
        {
            "asset_profiles": [{"asset_id": "box_anchor", "visual_role": "appearance_anchor", "quality_score": 88}],
            "assets": [
                {
                    "asset_id": "box_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/box.png",
                    "is_supported": True,
                    "visual_role": "appearance_anchor",
                    "quality_score": 88,
                }
            ],
        },
    )

    assert len(storyboard) == 3
    assert {shot["expression_plan_source"] for shot in storyboard} == {"fallback_rule_plan"}
    public_text = "".join(shot["subtitle"] + shot["voiceover"] for shot in storyboard)
    assert "看看这个好物" not in public_text
    assert "点击了解更多" not in public_text
    assert "真实外观确认" not in public_text
    assert "真实体验" not in public_text


def test_prompt_skill_template_loader_renders_markdown_variables():
    prompt = workflow._render_prompt_skill_template(
        "commerce_scene.source_confirm",
        {
            "product_type": "水杯",
            "appearance": "透明棕色杯身，蓝色环，黄色杯盖",
            "source_place": "玄关桌面",
            "style": "真实写实，自然晨光",
        },
        fallback="fallback {{product_type}}",
    )

    assert "上传素材中的同一件水杯" in prompt
    assert "透明棕色杯身，蓝色环，黄色杯盖" in prompt
    assert "玄关桌面" in prompt
    assert "{{" not in prompt
    assert "fallback" not in prompt


def test_prompt_skill_template_extractor_stops_before_next_section():
    template = workflow._extract_prompt_skill_template(
        """
---
id: demo
---
# Demo

## Prompt 模板

第一帧展示{{product_type}}。

## 校验规则

这段内部规则不能进入视频 prompt。
"""
    )
    prompt = workflow._render_prompt_skill_template(
        "missing.skill",
        {"product_type": "水杯"},
        template,
    )

    assert prompt == "第一帧展示水杯。"
    assert "校验规则" not in prompt
    assert "内部规则" not in prompt


def test_prompt_skill_template_loader_falls_back_when_template_missing(monkeypatch):
    monkeypatch.setitem(workflow._PROMPT_SKILL_TEMPLATE_CACHE, "commerce_scene.missing_template", "")

    prompt = workflow._render_prompt_skill_template(
        "commerce_scene.missing_template",
        {"product_type": "笔记本电脑"},
        "这是安全 fallback：{{product_type}} 仍然可见。",
    )

    assert prompt == "这是安全 fallback：笔记本电脑 仍然可见。"


def test_ideal_commerce_storyboard_records_per_shot_prompt_skill_ids():
    storyboard = workflow._plan_ideal_commerce_scene_storyboard(
        {
            "product_identity_card": {
                "product_type": "水杯",
                "appearance_summary": "透明棕色水杯，蓝色环和黄色杯盖",
                "visible_marks": ["CHAKO LAB 字样"],
            },
            "selling_points": ["便携", "大容量"],
            "usage_scene": "通勤",
        },
        {
            "asset_profiles": [{"asset_id": "cup_anchor", "visual_role": "appearance_anchor", "quality_score": 90}],
            "assets": [
                {
                    "asset_id": "cup_anchor",
                    "asset_type": "image",
                    "file_path": "/tmp/cup_anchor.png",
                    "is_supported": True,
                    "visual_role": "appearance_anchor",
                    "quality_score": 90,
                }
            ],
        },
    )

    assert [shot["selected_prompt_skill"] for shot in storyboard] == [
        "commerce_scene.source_confirm",
        "commerce_scene.material_action_proof",
        "commerce_scene.new_scene_result",
    ]
    assert "第一帧就是上传素材" in storyboard[0]["video_prompt"]
    assert "本镜头只发生一个主要动作" in storyboard[1]["video_prompt"]
    assert "硬切后的新分镜" in storyboard[2]["video_prompt"]
    assert all("不要新增非商品自带文字" in shot["video_prompt"] for shot in storyboard)


def _v3_laptop_product_context():
    return {
        "duration_seconds": 15,
        "target_platform": "tiktok",
        "product_title": "雷蛇笔记本",
        "selling_points": ["轻薄好收纳", "移动办公更省心", "高性能处理器"],
        "product_identity_card": {
            "product_type": "笔记本电脑",
            "identity_confidence": "high",
            "appearance_anchor_available": True,
            "appearance_summary": "黑色磨砂闭合笔记本，A 面有绿色蛇形标识。",
            "visible_marks": ["绿色蛇形 logo", "RAZER 标识"],
            "must_preserve": ["黑色闭合机身", "A 面绿色标识", "RAZER 标识"],
            "forbidden_changes": ["logo 重绘", "打开屏幕", "翻转"],
            "reference_asset_ids": ["laptop_anchor"],
        },
    }


def _v3_laptop_asset_analysis():
    return {
        "assets": [
            {
                "asset_id": "laptop_anchor",
                "asset_type": "image",
                "file_path": "/tmp/laptop_anchor.png",
                "is_supported": True,
                "visual_role": "appearance_anchor",
                "quality_score": 92,
            }
        ],
        "asset_profiles": [
            {
                "asset_id": "laptop_anchor",
                "visual_role": "appearance_anchor",
                "quality_score": 92,
                "product_visibility": "主体清晰",
                "normalized_roles": ["appearance_anchor_candidate"],
                "material_capabilities": {"appearance_anchor_candidate": True},
            }
        ],
        "product_identity_card": _v3_laptop_product_context()["product_identity_card"],
    }


def test_v3_template_storyboard_passes_semantic_narrative_review_without_hook_cta_roles():
    product_context = _v3_laptop_product_context()
    asset_analysis = _v3_laptop_asset_analysis()
    storyboard, script_plan = workflow.plan_storyboard_from_template(
        product_context,
        asset_analysis,
    )

    roles = [shot["narrative_role"] for shot in storyboard]
    assert roles == ["product_reveal", "feature_demo", "commerce_result"]
    assert "hook" not in roles
    assert "cta" not in roles

    review = workflow._rule_based_narrative_review(product_context, script_plan, storyboard)

    assert review["passed"] is True
    assert not any("缺少 hook" in issue or "缺少 cta" in issue for issue in review["issues"])


def test_v3_template_plan_records_strategy_contract_and_semantic_coverage():
    product_context = _v3_laptop_product_context()
    asset_analysis = _v3_laptop_asset_analysis()
    storyboard, script_plan = workflow.plan_storyboard_from_template(
        product_context,
        asset_analysis,
    )

    plan_contract = script_plan.get("plan_contract") or {}
    assert plan_contract["strategy_id"] == "product_fidelity_v3"
    assert plan_contract["strategy_family"] == "template_product_fidelity"
    assert plan_contract["expected_shape"]["shot_count"] == 3
    assert plan_contract["role_mapping"] == {
        "product_reveal": ["attention", "identity"],
        "feature_demo": ["proof"],
        "commerce_result": ["result", "conversion_intent"],
    }
    assert set(plan_contract["required_coverage"]) == {
        "attention",
        "identity",
        "proof",
        "result",
        "conversion_intent",
    }
    assert script_plan["_source"].startswith("product_fidelity_v3")
    assert all(str(shot.get("planner_source", "")).startswith("product_fidelity_v3") for shot in storyboard)


def test_forced_video_prompt_reaches_renderer_without_structured_constraint_appendix():
    forced_prompt = (
        "这是 5 秒文生视频，是硬切后的新镜头。第一秒建立办公室入口，"
        "第二到四秒人物整理背包侧袋里的同一只水杯，最后一秒停在商品结果画面。"
    )
    shot = {
        "shot_index": 1,
        "duration_seconds": 5,
        "render_strategy": "text_to_video",
        "force_video_prompt": True,
        "video_prompt": forced_prompt,
        "video_prompt_constraints": {
            "must_preserve": ["不应追加到最终 prompt"],
            "must_avoid": ["不应追加到最终 prompt"],
        },
        "product_identity_card": {
            "appearance_summary": "不应追加到最终 prompt",
            "visible_marks": ["不应追加到最终 prompt"],
            "must_preserve": ["不应追加到最终 prompt"],
            "forbidden_changes": ["不应追加到最终 prompt"],
        },
        "forbidden_variation": ["不应追加到最终 prompt"],
        "visual_style_bible": {"realism": "不应追加到最终 prompt"},
        "scene_goal": "不应追加到最终 prompt",
        "action": "不应追加到最终 prompt",
    }

    prompt = _build_seedance_prompt(shot)
    payload = _build_seedance_payload("doubao-seedance-test", shot)
    payload_prompt = payload["content"][0]["text"]

    assert prompt == forced_prompt
    assert payload_prompt == forced_prompt
    for leaked_fragment in (
        "product_identity_card",
        "video_prompt_constraints",
        "forbidden_variation",
        "不应追加到最终 prompt",
        "商品身份约束",
        "必须避免",
        "整体风格",
    ):
        assert leaked_fragment not in payload_prompt


def test_v3_default_product_fidelity_path_does_not_emit_fallback_or_diagnostic_subtitles():
    product_context = _v3_laptop_product_context()
    asset_analysis = _v3_laptop_asset_analysis()
    storyboard, script_plan = workflow.plan_storyboard_from_template(
        product_context,
        asset_analysis,
    )

    forbidden_public_lines = {"看看这个好物", "点击了解更多", "点击查看详情", "真实外观确认"}
    visible_lines = set()
    visible_lines.update(str(shot.get("subtitle", "")).strip() for shot in storyboard)
    visible_lines.update(str(shot.get("voiceover", "")).strip() for shot in storyboard)
    visible_lines.update(str(beat.get("subtitle", "")).strip() for beat in script_plan.get("beats", []))
    visible_lines.add(str(script_plan.get("hook", "")).strip())
    visible_lines.add(str(script_plan.get("cta", "")).strip())
    visible_lines.add(str(script_plan.get("full_subtitle_script", "")).strip())

    assert not forbidden_public_lines & visible_lines
    assert "保守降级结构" not in str(script_plan.get("style_notes", ""))
