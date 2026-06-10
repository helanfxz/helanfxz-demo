"""Deterministic final checks for generated video drafts."""

from __future__ import annotations

from typing import Any


def run_final_check(
    product_context: dict[str, Any],
    storyboard: list[dict[str, Any]],
    creation_plan: dict[str, Any],
    render_result: dict[str, Any],
    asset_gap_completion: dict[str, Any] | None = None,
    content_review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check whether a draft plan is eligible to be shown as ready."""

    issues: list[str] = []
    expected_duration = int(product_context.get("duration_seconds", 15))
    actual_duration = int(creation_plan.get("total_duration_seconds", 0))

    if actual_duration <= 0:
        issues.append("视频总时长无效。")
    if actual_duration > max(expected_duration, 15):
        issues.append("分镜总时长超过预期。")
    if not storyboard:
        issues.append("没有生成分镜。")
    if not render_result.get("success"):
        issues.append(f"预览视频生成失败：{render_result.get('error')}")
    if render_result.get("fallback_from"):
        fallback_error = render_result.get("fallback_from", {}).get("error", "unknown error")
        issues.append(f"Seedance fallback used; local preview needs review: {fallback_error}")
    if content_review and not content_review.get("passed", True):
        issues.append(f"内容审视未通过：{content_review.get('summary', content_review.get('error', '需要人工确认'))}")
    if asset_gap_completion and int(asset_gap_completion.get("unresolved_count", 0) or 0) > 0:
        issues.append(f"存在 {asset_gap_completion.get('unresolved_count')} 个商品主镜头缺少真实素材，已阻断文生视频硬生成，需要补充素材或人工确认降级。")
    for shot in storyboard:
        if not shot.get("visual_description"):
            issues.append(f"第 {shot.get('shot_index')} 个分镜缺少画面描述。")
        if not shot.get("subtitle"):
            issues.append(f"第 {shot.get('shot_index')} 个分镜缺少字幕。")

    return {
        "passed": not issues,
        "issues": issues,
        "actual_duration_seconds": actual_duration,
        "expected_duration_seconds": expected_duration,
    }
