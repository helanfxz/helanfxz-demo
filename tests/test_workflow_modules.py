from agent.content_repair import repair_rendered_content
from agent.final_checks import run_final_check


def test_final_checks_module_reports_empty_storyboard():
    result = run_final_check(
        product_context={"duration_seconds": 15},
        storyboard=[],
        creation_plan={"total_duration_seconds": 0},
        render_result={"success": False, "error": "missing video"},
    )

    assert result["passed"] is False
    assert "没有生成分镜。" in result["issues"]
    assert "视频总时长无效。" in result["issues"]


def test_content_repair_module_returns_empty_summary_without_records(tmp_path):
    result = repair_rendered_content(
        task_id="task-1",
        repair_records=[],
        creation_plan={"shots": []},
        render_result={},
        output_dir=str(tmp_path),
        report=lambda *args: None,
        repair_func=lambda **kwargs: {"success": True},
        flow_print=lambda message: None,
    )

    assert result == {
        "attempted_count": 0,
        "succeeded_count": 0,
        "failed_count": 0,
        "skipped_count": 0,
        "reconcat_success": False,
        "records": [],
    }
