from __future__ import annotations

from pathlib import Path
from typing import Any


PROMPT_SKILL_LIBRARY_DIR = Path(__file__).resolve().parents[1] / "prompt_skill_library"
REQUIRED_FRONT_MATTER = ("id", "strategy", "required_slots", "forbidden_if", "failure_tags", "success_stats")

_PRODUCT_WORDS = ("商品", "水杯", "杯", "笔记本", "电脑", "香水", "护肤", "背包", "服饰", "小家电")
_NEW_SCENE_WORDS = ("新地点", "新场景", "户外", "公园", "办公室入口", "写字楼", "通勤", "地铁", "长椅")


def _skill_path(skill_id: str) -> Path:
    parts = [part for part in str(skill_id or "").strip().split(".") if part]
    if len(parts) < 2:
        return PROMPT_SKILL_LIBRARY_DIR / "_invalid_skill_id_.md"
    return PROMPT_SKILL_LIBRARY_DIR.joinpath(*parts[:-1], f"{parts[-1]}.md")


def _parse_front_matter(text: str) -> dict[str, Any]:
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end].strip("\n")
    data: dict[str, Any] = {}
    current_key = ""
    for raw_line in block.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if not raw_line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            current_key = key.strip()
            data[current_key] = value.strip()
            continue
        if line.strip().startswith("- ") and current_key:
            existing = data.get(current_key)
            if not isinstance(existing, list):
                existing = []
                data[current_key] = existing
            existing.append(line.strip()[2:].strip())
    return data


def load_prompt_skill_contract(skill_id: str) -> dict[str, Any]:
    path = _skill_path(skill_id)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {"id": skill_id, "path": str(path), "exists": False, "front_matter": {}, "text": ""}
    return {
        "id": skill_id,
        "path": str(path),
        "exists": True,
        "front_matter": _parse_front_matter(text),
        "text": text,
    }


def validate_prompt_skill_contract(contract: dict[str, Any]) -> list[str]:
    if not contract.get("exists"):
        return [f"skill file missing: {contract.get('path')}"]
    front_matter = contract.get("front_matter") if isinstance(contract.get("front_matter"), dict) else {}
    issues: list[str] = []
    for field in REQUIRED_FRONT_MATTER:
        if field not in front_matter:
            issues.append(f"missing front matter field: {field}")
            continue
        value = front_matter.get(field)
        if field != "success_stats" and value in ("", None, []):
            issues.append(f"missing front matter field: {field}")
    if front_matter.get("id") and front_matter.get("id") != contract.get("id"):
        issues.append(f"id mismatch: expected {contract.get('id')}, got {front_matter.get('id')}")
    if "## Prompt 模板" not in str(contract.get("text", "")):
        issues.append("missing section: ## Prompt 模板")
    return issues


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def validate_shot_skill_contract(shot: dict[str, Any]) -> list[str]:
    skill_id = str(shot.get("selected_prompt_skill", "")).strip()
    render_strategy = str(shot.get("render_strategy", "")).strip()
    product_presence = str(shot.get("product_presence", "")).strip().lower()
    prompt = str(shot.get("video_prompt") or shot.get("visual_description") or "")
    issues: list[str] = []

    if skill_id == "commerce_scene.new_scene_result":
        if render_strategy != "text_to_video":
            issues.append("commerce_scene.new_scene_result requires render_strategy=text_to_video")
        if product_presence == "forbidden" and _contains_any(prompt, _PRODUCT_WORDS):
            issues.append("product_presence=forbidden conflicts with product-reconstruction prompt")

    if skill_id.startswith("source_scene_extension."):
        if render_strategy != "image_to_video":
            issues.append("source_scene_extension skill requires render_strategy=image_to_video")
        if product_presence != "required":
            issues.append("source_scene_extension skill requires product_presence=required")
        if _contains_any(prompt, _NEW_SCENE_WORDS) and ("第一帧" in prompt or "首帧" in prompt):
            issues.append("source-frame skill requests a new scene while using source first frame")

    return issues
