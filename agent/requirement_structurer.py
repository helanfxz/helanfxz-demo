from __future__ import annotations

import builtins
import json
import os
import re
from typing import Any

from agent.video_generation_workflow import _call_text_llm, _extract_json_from_text

VERBOSE_LOG = os.getenv("AIGC_VERBOSE_LOG") == "1"


def print(*args, **kwargs):  # type: ignore[override]
    if VERBOSE_LOG:
        builtins.print(*args, **kwargs)


def _flow_print(message: str) -> None:
    builtins.print(message, flush=True)


def _effective_chars(text: str) -> int:
    cleaned = re.sub(r"\s+", "", text)
    return len(cleaned)


def check_input_sufficiency(task_data: dict[str, Any]) -> dict[str, Any]:
    parts = [
        str(task_data.get("title", "")),
        " ".join(str(sp) for sp in task_data.get("selling_points", [])),
        str(task_data.get("target_audience", "")),
        str(task_data.get("usage_scene", "")),
        str(task_data.get("creative_direction", "")),
        str(task_data.get("custom_style_prompt", "")),
        str(task_data.get("product_type", "")),
        " ".join(str(item) for item in task_data.get("forbidden_changes", [])),
        " ".join(str(msg) for msg in task_data.get("chat_history", [])),
    ]
    total_chars = _effective_chars("".join(parts))
    conditions: list[str] = []

    if total_chars >= 60:
        confidence = "high"
    elif total_chars >= 30:
        confidence = "medium"
    else:
        confidence = "low"
        conditions.append(f"当前文字信息总量只有 {total_chars} 个有效字符，少于 30 个")

    if confidence == "high":
        warning_message = ""
    elif confidence == "medium":
        warning_message = "输入信息基本可用，但补充目标人群、使用场景或不允许变化项会提升生成稳定性。"
    else:
        warning_message = f"输入信息严重不足：{'；'.join(conditions)}。强烈建议补充后再生成，否则效果可能不理想。"

    return {
        "score": 1 if confidence == "low" else 0,
        "conditions": conditions,
        "confidence": confidence,
        "warning_message": warning_message,
    }

def structure_requirements(
    task_data: dict[str, Any],
    chat_history: list[str] | None = None,
) -> dict[str, Any]:
    if chat_history is None:
        chat_history = task_data.get("chat_history", [])

    prompt = {
        "任务": "请根据以下表单信息、选项和对话历史，整理出一份结构化的需求摘要。",
        "表单信息": {
            "标题": task_data.get("title", ""),
            "卖点": task_data.get("selling_points", []),
            "目标受众": task_data.get("target_audience", ""),
            "使用场景": task_data.get("usage_scene", ""),
            "创意方向": task_data.get("creative_direction", ""),
            "风格": task_data.get("style", ""),
            "自定义风格提示": task_data.get("custom_style_prompt", ""),
            "禁止变更": task_data.get("forbidden_changes", []),
            "商品类型": task_data.get("product_type", ""),
        },
        "对话历史": chat_history,
        "输出格式": {
            "target_audience": "目标受众，综合表单和对话提炼",
            "usage_scene": "使用场景，综合表单和对话提炼",
            "creative_goal": "创意目标，一句话概括本次视频要达成的效果",
            "selling_point_priority": "卖点优先级列表，按重要性从高到低排列",
            "must_preserve": "必须保留的元素（品牌名、核心卖点关键词、禁止变更项等）",
            "avoid": "需要避免的内容或风格",
            "tone": "语气风格（如专业、活泼、温馨、高端等）",
            "extra_requirements": "从对话中提取的额外需求",
            "input_confidence": "high / medium / low，根据信息充分程度判断",
        },
        "要求": "严格按上述 JSON 格式输出，不要添加多余文本。中文输出。",
    }

    llm_result = _call_text_llm(prompt, purpose="requirement_structuring")

    if llm_result["ok"]:
        parsed = _extract_json_from_text(llm_result["content"])
        if parsed and isinstance(parsed, dict):
            structured = _normalize_structured_requirements(parsed)
            structured["llm_enabled"] = True
            _flow_print("[requirement_structurer] LLM 需求结构化完成")
            return structured

    _flow_print("[requirement_structurer] LLM 不可用，使用规则降级提取")
    return _fallback_structure_requirements(task_data, chat_history)

def _normalize_structured_requirements(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_audience": str(parsed.get("target_audience", "")),
        "usage_scene": str(parsed.get("usage_scene", "")),
        "creative_goal": str(parsed.get("creative_goal", "")),
        "selling_point_priority": (
            parsed.get("selling_point_priority")
            if isinstance(parsed.get("selling_point_priority"), list)
            else []
        ),
        "must_preserve": (
            parsed.get("must_preserve")
            if isinstance(parsed.get("must_preserve"), list)
            else []
        ),
        "avoid": (
            parsed.get("avoid")
            if isinstance(parsed.get("avoid"), list)
            else []
        ),
        "tone": str(parsed.get("tone", "")),
        "extra_requirements": str(parsed.get("extra_requirements", "")),
        "input_confidence": (
            parsed["input_confidence"]
            if parsed.get("input_confidence") in ("high", "medium", "low")
            else "medium"
        ),
    }


def _fallback_structure_requirements(
    task_data: dict[str, Any],
    chat_history: list[str],
) -> dict[str, Any]:
    selling_points = task_data.get("selling_points", [])
    forbidden_changes = task_data.get("forbidden_changes", [])

    must_preserve: list[str] = []
    title = task_data.get("title", "").strip()
    if title:
        must_preserve.append(title)
    must_preserve.extend(str(fc) for fc in forbidden_changes if str(fc).strip())

    sufficiency = check_input_sufficiency(task_data)

    extra_parts: list[str] = []
    for msg in chat_history:
        stripped = str(msg).strip()
        if stripped and _effective_chars(stripped) >= 4:
            extra_parts.append(stripped)

    return {
        "target_audience": task_data.get("target_audience", "").strip(),
        "usage_scene": task_data.get("usage_scene", "").strip(),
        "creative_goal": "",
        "selling_point_priority": [str(sp).strip() for sp in selling_points if str(sp).strip()],
        "must_preserve": must_preserve,
        "avoid": [],
        "tone": task_data.get("style", "").strip(),
        "extra_requirements": "\n".join(extra_parts) if extra_parts else "",
        "input_confidence": sufficiency["confidence"],
        "llm_enabled": False,
    }

_QUESTION_TEMPLATES: list[dict[str, str]] = [
    {
        "field_target": "target_audience",
        "question": "这个视频主要想吸引哪类人群？比如：年轻女性、宝妈、职场白领、学生党等。明确受众能帮我更精准地选择表达方式。",
    },
    {
        "field_target": "usage_scene",
        "question": "用户通常在什么场景下使用这个产品？比如：日常居家、户外旅行、办公室、送礼等。场景越具体，画面越有代入感。",
    },
    {
        "field_target": "selling_point_priority",
        "question": "这些卖点里，哪个是用户最关心的？如果只能突出一个，你会选哪个？这决定了视频的核心信息层级。",
    },
    {
        "field_target": "must_preserve",
        "question": "有没有绝对不能改动的元素？比如品牌名、特定包装颜色、标志性口号等。提前告诉我，可以避免生成偏差。",
    },
    {
        "field_target": "creative_direction",
        "question": "你希望视频的整体感觉是什么？比如：高端大气、亲切温馨、活力潮流、专业可信等。这会影响画面风格和节奏。",
    },
]


def generate_chat_question(
    task_data: dict[str, Any],
    chat_history: list[str] | None = None,
    question_index: int = 0,
) -> dict[str, Any]:
    if chat_history is None:
        chat_history = task_data.get("chat_history", [])

    filled_fields: set[str] = set()
    if task_data.get("target_audience", "").strip():
        filled_fields.add("target_audience")
    if task_data.get("usage_scene", "").strip():
        filled_fields.add("usage_scene")
    if task_data.get("creative_direction", "").strip():
        filled_fields.add("creative_direction")
    if task_data.get("selling_points", []):
        filled_fields.add("selling_point_priority")
    if task_data.get("forbidden_changes", []):
        filled_fields.add("must_preserve")

    chat_text = "\n".join(str(m) for m in chat_history)
    if "受众" in chat_text or "人群" in chat_text or "用户是" in chat_text:
        filled_fields.add("target_audience")
    if "场景" in chat_text or "什么时候用" in chat_text:
        filled_fields.add("usage_scene")
    if "卖点" in chat_text and ("最" in chat_text or "优先" in chat_text or "核心" in chat_text):
        filled_fields.add("selling_point_priority")
    if "不能改" in chat_text or "保留" in chat_text or "禁止" in chat_text:
        filled_fields.add("must_preserve")
    if "风格" in chat_text or "感觉" in chat_text or "调性" in chat_text:
        filled_fields.add("creative_direction")

    unfilled = [qt for qt in _QUESTION_TEMPLATES if qt["field_target"] not in filled_fields]

    if not unfilled:
        return {
            "question": "感谢你的补充！信息已经很充分了，我现在可以开始为你生成视频方案。",
            "field_target": "complete",
        }

    actual_index = question_index % len(unfilled)
    selected = unfilled[actual_index]

    return {
        "question": selected["question"],
        "field_target": selected["field_target"],
    }
