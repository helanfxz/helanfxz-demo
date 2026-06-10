import agent.seedance_video_renderer as renderer


class _FakeSubtitleFont:
    def __init__(self, size: int = 44):
        self.size = size


class _FakeSubtitleDraw:
    def textbbox(self, _xy, text, font):
        return (0, 0, len(str(text)) * font.size, font.size)


def test_required_failed_shot_prevents_successful_partial_concat(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK_API_KEY", "test-key")
    monkeypatch.setenv("ARK_VIDEO_ENDPOINT_ID", "test-endpoint")
    monkeypatch.setattr(renderer, "_split_seedance_render_batches", lambda indexed: [indexed])
    monkeypatch.setattr(
        renderer,
        "_render_seedance_batch",
        lambda *args, **kwargs: [
            {"shot_index": 0, "success": True, "video_url": "http://example/0.mp4", "seedance_task_id": "ok0"},
            {"shot_index": 1, "success": True, "video_url": "http://example/1.mp4", "seedance_task_id": "ok1"},
            {"shot_index": 2, "success": False, "error": "Invalid content.text", "seedance_task_id": "bad2"},
        ],
    )
    monkeypatch.setattr(
        renderer,
        "_download_video",
        lambda url, path: (path.write_bytes(b"fake"), {"success": True, "path": str(path), "error": None})[1],
    )
    monkeypatch.setattr(renderer, "_adapt_clip_to_target_duration", lambda source_path, output_dir, shot_index, shot: source_path)
    monkeypatch.setattr(renderer, "_concat_videos", lambda *args, **kwargs: {"success": True, "error": None})
    monkeypatch.setattr(renderer, "_overlay_storyboard_subtitles", lambda *args, **kwargs: {"success": True, "error": None})

    result = renderer.render_seedance_video(
        task_id="task-partial",
        creation_plan={
            "shots": [
                {"shot_index": 0, "product_presence": "required", "required_for_variant": True},
                {"shot_index": 1, "product_presence": "required", "required_for_variant": True},
                {"shot_index": 2, "product_presence": "optional", "required_for_variant": True},
            ]
        },
        output_dir=str(tmp_path),
    )

    assert result["success"] is False
    assert "required shot failed" in result["error"]


def test_subtitle_width_wrapper_keeps_complete_reasonable_caption():
    font = _FakeSubtitleFont(12)
    draw = _FakeSubtitleDraw()
    caption = "透明杯身一眼看清饮水余量，放在桌边也更安心"

    lines = renderer._wrap_plain_subtitle_text_by_width(
        draw,
        caption,
        font,
        max_width=120,
        max_lines=5,
    )

    assert "".join(lines) == caption
    assert len(lines) > 1
    assert all(renderer._text_width(draw, line, font) <= 120 for line in lines)
