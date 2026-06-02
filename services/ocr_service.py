"""
OCR text extraction service for multimodal retrieval.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from config.config_manager import get_config
from monitoring.logger import get_logger

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:  # pragma: no cover
    RapidOCR = None


TOKEN_PATTERN = re.compile(r"[A-Za-z]+[A-Za-z0-9._-]*|\d+(?:[./-]\d+)*|[\u4e00-\u9fff]+")
IDENTIFIER_PATTERN = re.compile(r"^(?=.*\d)[A-Za-z0-9][A-Za-z0-9._/-]{3,}$")
LONG_NUMBER_PATTERN = re.compile(r"^\d{4,}(?:[./-]\d+)*$")
RATIO_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*[:：]\s*\d+(?:\.\d+)?")
MATERIAL_PATTERN = re.compile(
    r"(?:HT\d+|Q\d+[A-Z]?|QT\d+|ZG\d+|[0-9]+#钢|[0-9]+mn|[0-9]+Mn|A3钢|A3|45#|65Mn|45钢|铸铁|钢|铜|铝)",
    re.IGNORECASE,
)
IGNORE_FIELD_LINES = (
    "技术要求",
    "未注",
    "倒角",
    "学校",
    "学院",
    "大学",
    "工学院",
    "制图",
    "审核",
    "校核",
    "批准",
    "日期",
)
SHEET_SIZE_VALUES = {"a0", "a1", "a2", "a3", "a4"}
COMMON_STOPWORDS = {
    "图号",
    "比例",
    "材料",
    "数量",
    "设计",
    "审核",
    "备注",
    "技术要求",
    "年月",
    "件数",
    "序号",
    "名称",
    "规格",
    "单位",
    "共",
    "第",
    "张",
    "图",
    "其余",
}
FIELD_LABEL_ALIASES = {
    "drawing_no": ("图号", "图样标记", "代号", "编号"),
    "part_name": ("零件名称", "零件名", "名称", "图名"),
    "material": ("材料", "材质"),
    "scale": ("比例", "比列"),
    "quantity": ("数量", "件数"),
}
FIELD_LABEL_PRIORITY = {
    "drawing_no": 1.0,
    "part_name": 0.95,
    "material": 0.85,
    "scale": 0.80,
    "quantity": 0.75,
}
FIELD_VALUE_MAXLEN = {
    "drawing_no": 32,
    "part_name": 24,
    "material": 16,
    "scale": 12,
    "quantity": 8,
}
INVALID_DRAWING_NO_VALUES = {
    "图号",
    "名称",
    "序号",
    "序号名称",
    "图样标记",
    "代号",
    "规格",
    "单位",
    "标记",
}
INVALID_PART_NAME_VALUES = {
    "名称",
    "序号",
    "序号名称",
    "规格",
    "图号",
    "图样标记",
    "代号",
    "单位",
    "校核",
    "审核",
    "批准",
    "设计",
    "备注",
    "标准备注",
}


def normalize_text(value: str) -> str:
    text = value.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_filename_text(filename: str) -> str:
    stem = Path(str(filename or "")).stem
    stem = re.sub(r"\.(dwg|dxf|step|stp|pdf)\b", " ", stem, flags=re.IGNORECASE)
    stem = re.sub(r"-\d{4}-\d{2}-\d{2}(?:-\d{2}){3}-\d+", " ", stem)
    stem = re.sub(r"[_\-]+", " ", stem)
    stem = re.sub(r"\s+", " ", stem).strip()
    return stem


def estimate_text_quality(
    text_lines: List[str],
    tokens: List[str],
    number_tokens: List[str],
    identifier_tokens: List[str],
) -> float:
    if not text_lines:
        return 0.0
    line_score = min(1.0, len(text_lines) / 4.0)
    token_score = min(1.0, len(tokens) / 8.0)
    number_score = min(1.0, len(number_tokens) / 4.0)
    identifier_score = min(1.0, len(identifier_tokens) / 3.0)
    quality = 0.20 * line_score + 0.25 * token_score + 0.20 * number_score + 0.35 * identifier_score
    return float(max(0.0, min(1.0, quality)))


def build_text_descriptor(text_lines: List[str], max_text_length: int = 500) -> Dict:
    filtered_lines = [normalize_text(line) for line in text_lines if normalize_text(line)]
    raw_text = " ".join(filtered_lines)[:max_text_length]
    tokens = TOKEN_PATTERN.findall(raw_text)
    normalized_tokens = []
    for token in tokens:
        lowered = token.lower()
        if len(lowered) == 1 and not any(ch.isdigit() for ch in lowered):
            continue
        if lowered in COMMON_STOPWORDS:
            continue
        normalized_tokens.append(lowered)
    unique_tokens = sorted(set(normalized_tokens))[:48]
    number_tokens = sorted({token for token in unique_tokens if any(ch.isdigit() for ch in token)})[:24]
    identifier_tokens = sorted(
        {
            token
            for token in unique_tokens
            if IDENTIFIER_PATTERN.match(token) or LONG_NUMBER_PATTERN.match(token)
        }
    )[:24]

    chars = re.sub(r"\s+", "", raw_text.lower())
    chargrams = []
    for n in (2, 3):
        if len(chars) >= n:
            chargrams.extend(chars[i : i + n] for i in range(len(chars) - n + 1))
    unique_chargrams = sorted(set(chargrams))[:160]

    return {
        "raw_text": raw_text,
        "text_lines": filtered_lines[:12],
        "tokens": unique_tokens,
        "number_tokens": number_tokens,
        "identifier_tokens": identifier_tokens,
        "chargrams": unique_chargrams,
        "has_text": bool(raw_text),
        "text_quality": estimate_text_quality(filtered_lines, unique_tokens, number_tokens, identifier_tokens),
    }


def _cleanup_field_value(value: str) -> str:
    cleaned = normalize_text(value)
    cleaned = cleaned.replace("：", ":")
    cleaned = re.sub(r"^[：:\-_\s]+", "", cleaned)
    return cleaned.strip()


def _empty_title_block_fields() -> Dict:
    return {
        "part_name": "",
        "drawing_no": "",
        "material": "",
        "scale": "",
        "quantity": "",
        "field_quality": 0.0,
        "kv_pair_count": 0,
        "detection_method": "rules",
        "field_sources": {
            "part_name": "",
            "drawing_no": "",
            "material": "",
            "scale": "",
            "quantity": "",
        },
    }


def _normalize_label_text(value: str) -> str:
    return re.sub(r"[\s:：\-_./|()\[\]【】<>]+", "", str(value or ""))


def _extract_box_coords(box) -> Optional[Tuple[float, float, float, float]]:
    try:
        points = np.asarray(box, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return None
    if points.size == 0:
        return None
    x1 = float(points[:, 0].min())
    y1 = float(points[:, 1].min())
    x2 = float(points[:, 0].max())
    y2 = float(points[:, 1].max())
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _serialize_ocr_entries(entries: List[Dict]) -> List[Dict]:
    serialized = []
    for entry in entries[:24]:
        serialized.append(
            {
                "text": entry.get("text", ""),
                "score": round(float(entry.get("score", 0.0)), 4),
                "x1": round(float(entry.get("x1", 0.0)), 4),
                "y1": round(float(entry.get("y1", 0.0)), 4),
                "x2": round(float(entry.get("x2", 0.0)), 4),
                "y2": round(float(entry.get("y2", 0.0)), 4),
                "cx": round(float(entry.get("cx", 0.0)), 4),
                "cy": round(float(entry.get("cy", 0.0)), 4),
                "width": round(float(entry.get("width", 0.0)), 4),
                "height": round(float(entry.get("height", 0.0)), 4),
                "source": entry.get("source", "unknown"),
            }
        )
    return serialized


def _label_key_from_text(text: str) -> Optional[str]:
    normalized = _normalize_label_text(text)
    if not normalized:
        return None
    for key, aliases in FIELD_LABEL_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalize_label_text(alias)
            if normalized == alias_norm or normalized.startswith(alias_norm) or alias_norm in normalized:
                return key
    return None


def _coerce_field_value(key: str, value: str) -> str:
    cleaned = _cleanup_field_value(value)
    if not cleaned:
        return ""
    cleaned = cleaned[: FIELD_VALUE_MAXLEN.get(key, 32)]
    lowered = cleaned.lower()

    if key == "drawing_no":
        condensed = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff./_-]", "", cleaned)
        if condensed in INVALID_DRAWING_NO_VALUES:
            return ""
        if lowered in SHEET_SIZE_VALUES:
            return ""
        material_match = MATERIAL_PATTERN.fullmatch(cleaned)
        if material_match:
            return ""
        tokens = TOKEN_PATTERN.findall(cleaned)
        identifier_candidates = [
            token
            for token in tokens
            if (IDENTIFIER_PATTERN.match(token) or LONG_NUMBER_PATTERN.match(token))
            and token.lower() not in SHEET_SIZE_VALUES
        ]
        if identifier_candidates:
            return _cleanup_field_value(identifier_candidates[0])
        if any(ch.isdigit() for ch in cleaned) and len(cleaned) >= 4:
            return cleaned
        return ""

    if key == "material":
        match = MATERIAL_PATTERN.search(cleaned)
        return _cleanup_field_value(match.group(0)) if match else ""

    if key == "scale":
        match = RATIO_PATTERN.search(cleaned)
        return _cleanup_field_value(match.group(0)) if match else ""

    if key == "quantity":
        tokens = [token for token in TOKEN_PATTERN.findall(cleaned) if any(ch.isdigit() for ch in token)]
        if tokens:
            return _cleanup_field_value(tokens[0])
        return ""

    if key == "part_name":
        condensed = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", cleaned)
        if condensed in INVALID_PART_NAME_VALUES:
            return ""
        if any(marker in cleaned for marker in IGNORE_FIELD_LINES):
            return ""
        if "备注" in cleaned:
            return ""
        cleaned = re.sub(r"^(零件名称|零件名|名称|图名)[:：]?", "", cleaned)
        condensed = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", cleaned)
        if MATERIAL_PATTERN.search(cleaned) or RATIO_PATTERN.search(cleaned):
            return ""
        if 2 <= len(condensed) <= 16 and re.search(r"[\u4e00-\u9fff]{2,}", condensed):
            return cleaned
        return ""

    return cleaned


def _extract_inline_field_value(key: str, text: str) -> str:
    candidate = str(text or "")
    for alias in FIELD_LABEL_ALIASES.get(key, ()):
        candidate = candidate.replace(alias, " ")
    candidate = re.sub(r"[：:]", " ", candidate)
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return _coerce_field_value(key, candidate)


def _vertical_overlap(a: Dict, b: Dict) -> float:
    top = max(float(a.get("y1", 0.0)), float(b.get("y1", 0.0)))
    bottom = min(float(a.get("y2", 0.0)), float(b.get("y2", 0.0)))
    overlap = max(0.0, bottom - top)
    denom = max(1e-6, min(float(a.get("height", 0.0)), float(b.get("height", 0.0))))
    return overlap / denom


def _horizontal_overlap(a: Dict, b: Dict) -> float:
    left = max(float(a.get("x1", 0.0)), float(b.get("x1", 0.0)))
    right = min(float(a.get("x2", 0.0)), float(b.get("x2", 0.0)))
    overlap = max(0.0, right - left)
    denom = max(1e-6, min(float(a.get("width", 0.0)), float(b.get("width", 0.0))))
    return overlap / denom


def _build_field_candidate_from_label(
    key: str,
    label_entry: Dict,
    entries: List[Dict],
) -> Tuple[str, float]:
    inline_value = _extract_inline_field_value(key, label_entry.get("text", ""))
    if inline_value:
        return inline_value, 1.2 + 0.2 * float(label_entry.get("score", 0.0))

    best_value = ""
    best_score = 0.0
    label_x2 = float(label_entry.get("x2", 0.0))
    label_y2 = float(label_entry.get("y2", 0.0))
    label_width = max(1e-6, float(label_entry.get("width", 0.0)))
    label_height = max(1e-6, float(label_entry.get("height", 0.0)))

    for candidate in entries:
        if candidate is label_entry:
            continue
        candidate_text = candidate.get("text", "")
        if _label_key_from_text(candidate_text):
            continue
        candidate_value = _coerce_field_value(key, candidate_text)
        if not candidate_value:
            continue

        dx = float(candidate.get("x1", 0.0)) - label_x2
        dy = float(candidate.get("y1", 0.0)) - label_y2
        row_overlap = _vertical_overlap(label_entry, candidate)
        col_overlap = _horizontal_overlap(label_entry, candidate)
        row_distance = abs(float(candidate.get("cy", 0.0)) - float(label_entry.get("cy", 0.0))) / label_height
        col_distance = abs(float(candidate.get("cx", 0.0)) - float(label_entry.get("cx", 0.0))) / label_width

        same_row = row_overlap >= 0.35 and dx >= -0.08 * label_width and dx <= 8.5 * label_width
        below_row = dy >= -0.15 * label_height and dy <= 3.5 * label_height and col_overlap >= 0.2
        if not same_row and not below_row:
            continue

        spatial_score = 0.0
        if same_row:
            spatial_score += 1.05 - 0.12 * max(0.0, dx / label_width)
            spatial_score += 0.40 * row_overlap
            spatial_score -= 0.05 * row_distance
        if below_row:
            spatial_score = max(
                spatial_score,
                0.82 - 0.10 * max(0.0, dy / label_height) + 0.20 * col_overlap - 0.03 * col_distance,
            )

        spatial_score += 0.12 * float(candidate.get("score", 0.0))
        spatial_score += 0.08 * FIELD_LABEL_PRIORITY.get(key, 0.5)
        if len(candidate_value) <= FIELD_VALUE_MAXLEN.get(key, 32):
            spatial_score += 0.05

        if spatial_score > best_score:
            best_value = candidate_value
            best_score = spatial_score

    return best_value, best_score


def _extract_title_block_fields_from_entries(entries: List[Dict]) -> Dict:
    fields = _empty_title_block_fields()
    if not entries:
        return fields

    label_entries = []
    for entry in entries:
        label_key = _label_key_from_text(entry.get("text", ""))
        if label_key:
            label_entries.append((label_key, entry))

    matched_pairs = 0
    field_scores: Dict[str, float] = {}
    for key, label_entry in label_entries:
        candidate_value, candidate_score = _build_field_candidate_from_label(key, label_entry, entries)
        if candidate_value and candidate_score > field_scores.get(key, 0.0):
            fields[key] = candidate_value
            field_scores[key] = candidate_score
            fields["field_sources"][key] = "kv_detection"
            matched_pairs += 1

    if not fields["drawing_no"]:
        for entry in sorted(entries, key=lambda item: (-float(item.get("score", 0.0)), float(item.get("y1", 0.0)))):
            candidate = _coerce_field_value("drawing_no", entry.get("text", ""))
            if candidate:
                fields["drawing_no"] = candidate
                field_scores["drawing_no"] = max(field_scores.get("drawing_no", 0.0), 0.55)
                fields["field_sources"]["drawing_no"] = "entry_fallback"
                break

    if not fields["part_name"]:
        for entry in entries:
            candidate = _coerce_field_value("part_name", entry.get("text", ""))
            if candidate:
                fields["part_name"] = candidate
                field_scores["part_name"] = max(field_scores.get("part_name", 0.0), 0.45)
                fields["field_sources"]["part_name"] = "entry_fallback"
                break

    populated = sum(1 for key in ("part_name", "drawing_no", "material", "scale", "quantity") if fields[key])
    if populated:
        confidence = min(1.0, sum(field_scores.values()) / max(1.0, 1.35 * populated))
        density = min(1.0, len(label_entries) / 4.0)
        fields["field_quality"] = round(max(populated / 5.0, 0.65 * confidence + 0.35 * density), 4)
    fields["kv_pair_count"] = int(matched_pairs)
    fields["detection_method"] = "kv_detection" if matched_pairs else "entry_fallback"
    return fields


def _extract_title_block_fields_from_lines(lines: List[str]) -> Dict:
    normalized_lines = [_cleanup_field_value(line) for line in lines if _cleanup_field_value(line)]
    fields = _empty_title_block_fields()
    pending_label = ""
    quantity_from_label = False
    label_hits = 0

    def assign_if_empty(key: str, value: str, source: str):
        cleaned = _coerce_field_value(key, value)
        if cleaned and not fields[key]:
            fields[key] = cleaned
            fields["field_sources"][key] = source

    for index, line in enumerate(normalized_lines):
        if any(marker in line for marker in IGNORE_FIELD_LINES):
            continue

        if pending_label:
            candidate = _coerce_field_value(pending_label, line)
            if candidate:
                assign_if_empty(pending_label, candidate, "rules_label")
                if pending_label == "quantity":
                    quantity_from_label = True
                pending_label = ""
                continue

        if "图号" in line:
            label_hits += 1
            value = _cleanup_field_value(line.replace("图号", ""))
            if value:
                assign_if_empty("drawing_no", value, "rules_label")
            else:
                pending_label = "drawing_no"
            continue
        if "材料" in line:
            label_hits += 1
            match = MATERIAL_PATTERN.search(line)
            if match:
                assign_if_empty("material", match.group(0), "rules_label")
            else:
                pending_label = "material"
            continue
        if "比例" in line:
            label_hits += 1
            match = RATIO_PATTERN.search(line)
            if match:
                assign_if_empty("scale", match.group(0), "rules_label")
            else:
                pending_label = "scale"
            continue
        if "数量" in line or "件数" in line:
            label_hits += 1
            digits = [token for token in TOKEN_PATTERN.findall(line) if any(ch.isdigit() for ch in token)]
            if digits:
                assign_if_empty("quantity", digits[0], "rules_label")
                quantity_from_label = True
            else:
                pending_label = "quantity"
            continue
        if "名称" in line and not fields["part_name"]:
            label_hits += 1
            value = _cleanup_field_value(line.replace("名称", ""))
            if value:
                assign_if_empty("part_name", value, "rules_label")
            else:
                pending_label = "part_name"
            continue

        if not fields["scale"]:
            match = RATIO_PATTERN.search(line)
            if match:
                assign_if_empty("scale", match.group(0), "rules_fallback")

        if not fields["material"]:
            match = MATERIAL_PATTERN.search(line)
            if match:
                assign_if_empty("material", match.group(0), "rules_fallback")

        if not fields["drawing_no"]:
            tokens = TOKEN_PATTERN.findall(line)
            identifier_candidates = [token for token in tokens if IDENTIFIER_PATTERN.match(token) and len(token) >= 6]
            if identifier_candidates:
                assign_if_empty("drawing_no", identifier_candidates[0], "rules_fallback")

        if not fields["part_name"]:
            if re.search(r"[\u4e00-\u9fff]{2,}", line) and not any(marker in line for marker in ("图号", "材料", "比例", "数量", "件数")):
                if not any(marker in line for marker in IGNORE_FIELD_LINES):
                    chinese_only = re.sub(r"[^A-Za-z0-9\u4e00-\u9fff]", "", line)
                    if 2 <= len(chinese_only) <= 12 and not MATERIAL_PATTERN.search(line):
                        assign_if_empty("part_name", line, "rules_fallback")

        if index + 1 < len(normalized_lines) and not fields["part_name"]:
            next_line = normalized_lines[index + 1]
            if line in {"名称", "零件名称", "零件名"} and re.search(r"[\u4e00-\u9fff]{2,}", next_line):
                assign_if_empty("part_name", next_line, "rules_label")

    if fields["quantity"] and not quantity_from_label:
        fields["quantity"] = ""
    if fields["part_name"] and any(marker in fields["part_name"] for marker in IGNORE_FIELD_LINES):
        fields["part_name"] = ""
    populated = sum(1 for key in ("part_name", "drawing_no", "material", "scale", "quantity") if fields[key])
    label_quality = min(1.0, label_hits / 4.0)
    fields["field_quality"] = round(0.55 * (populated / 5.0) + 0.45 * label_quality, 4)
    fields["detection_method"] = "rules"
    return fields


def extract_title_block_fields(lines: List[str], ocr_entries: Optional[List[Dict]] = None) -> Dict:
    line_fields = _extract_title_block_fields_from_lines(lines)
    if not ocr_entries:
        return line_fields

    entry_fields = _extract_title_block_fields_from_entries(ocr_entries)
    merged = _empty_title_block_fields()
    source_priority = {
        "kv_detection": 4,
        "rules_label": 3,
        "entry_fallback": 2,
        "rules_fallback": 1,
        "": 0,
    }
    for key in ("part_name", "drawing_no", "material", "scale", "quantity"):
        entry_source = (entry_fields.get("field_sources") or {}).get(key, "")
        line_source = (line_fields.get("field_sources") or {}).get(key, "")
        if source_priority.get(entry_source, 0) > source_priority.get(line_source, 0):
            merged[key] = entry_fields.get(key, "")
            merged["field_sources"][key] = entry_source
        elif source_priority.get(entry_source, 0) < source_priority.get(line_source, 0):
            merged[key] = line_fields.get(key, "")
            merged["field_sources"][key] = line_source
        else:
            chosen_value = entry_fields.get(key) or line_fields.get(key) or ""
            chosen_source = entry_source if entry_fields.get(key) else line_source
            merged[key] = chosen_value
            merged["field_sources"][key] = chosen_source

    populated = sum(1 for key in ("part_name", "drawing_no", "material", "scale", "quantity") if merged[key])
    trusted_count = sum(
        1
        for key in ("part_name", "drawing_no", "material", "scale", "quantity")
        if merged["field_sources"].get(key) in {"kv_detection", "rules_label"}
    )
    merged["kv_pair_count"] = int(entry_fields.get("kv_pair_count", 0))
    merged["detection_method"] = (
        entry_fields.get("detection_method", "kv_detection")
        if entry_fields.get("field_quality", 0.0) >= line_fields.get("field_quality", 0.0)
        else line_fields.get("detection_method", "rules")
    )
    merged["field_quality"] = round(
        max(
            0.55 * (populated / 5.0) + 0.45 * (trusted_count / 5.0),
            min(1.0, 0.7 * float(entry_fields.get("field_quality", 0.0)) + 0.3 * float(line_fields.get("field_quality", 0.0))),
        ),
        4,
    )
    return merged


def compare_title_block_fields(query_fields: Optional[Dict], target_fields: Optional[Dict]) -> Dict:
    empty = {
        "field_score": 0.0,
        "drawing_no_similarity": 0.0,
        "part_name_similarity": 0.0,
        "material_similarity": 0.0,
        "scale_similarity": 0.0,
        "quantity_similarity": 0.0,
        "exact_drawing_no_match": 0.0,
        "field_quality": 0.0,
        "drawing_no_trusted": 0.0,
        "part_name_trusted": 0.0,
        "trusted_field_coverage": 0.0,
    }
    if not query_fields or not target_fields:
        return empty

    def norm(value: str) -> str:
        return re.sub(r"\s+", "", str(value or "").lower())

    def sim(a: str, b: str) -> float:
        value_a = str(a or "").strip()
        value_b = str(b or "").strip()
        if not value_a or not value_b:
            return 0.0
        if norm(value_a) == norm(value_b):
            return 1.0
        return SequenceMatcher(None, norm(value_a), norm(value_b)).ratio()

    drawing_sim = sim(query_fields.get("drawing_no", ""), target_fields.get("drawing_no", ""))
    part_name_sim = sim(query_fields.get("part_name", ""), target_fields.get("part_name", ""))
    material_sim = sim(query_fields.get("material", ""), target_fields.get("material", ""))
    scale_sim = sim(query_fields.get("scale", ""), target_fields.get("scale", ""))
    quantity_sim = sim(query_fields.get("quantity", ""), target_fields.get("quantity", ""))
    exact_match = 1.0 if drawing_sim >= 0.999 and drawing_sim > 0 else 0.0
    query_sources = query_fields.get("field_sources", {}) or {}
    target_sources = target_fields.get("field_sources", {}) or {}
    drawing_no_trusted = 1.0 if (
        query_sources.get("drawing_no") in {"kv_detection", "rules_label"}
        and target_sources.get("drawing_no") in {"kv_detection", "rules_label"}
    ) else 0.0
    part_name_trusted = 1.0 if (
        query_sources.get("part_name") in {"kv_detection", "rules_label"}
        and target_sources.get("part_name") in {"kv_detection", "rules_label"}
    ) else 0.0
    trusted_field_coverage = sum(
        1.0
        for key in ("drawing_no", "part_name", "material", "scale", "quantity")
        if query_sources.get(key) in {"kv_detection", "rules_label"}
        and target_sources.get(key) in {"kv_detection", "rules_label"}
    ) / 5.0
    field_quality = min(
        float(query_fields.get("field_quality", 0.0)),
        float(target_fields.get("field_quality", 0.0)),
    )
    field_score = (
        0.45 * drawing_sim
        + 0.25 * part_name_sim
        + 0.15 * material_sim
        + 0.10 * scale_sim
        + 0.05 * quantity_sim
    ) * max(0.3, field_quality)

    return {
        "field_score": float(max(0.0, min(1.0, field_score))),
        "drawing_no_similarity": float(drawing_sim),
        "part_name_similarity": float(part_name_sim),
        "material_similarity": float(material_sim),
        "scale_similarity": float(scale_sim),
        "quantity_similarity": float(quantity_sim),
        "exact_drawing_no_match": float(exact_match),
        "field_quality": float(field_quality),
        "drawing_no_trusted": float(drawing_no_trusted),
        "part_name_trusted": float(part_name_trusted),
        "trusted_field_coverage": float(trusted_field_coverage),
    }


def _max_sequence_similarity(values_a: List[str], values_b: List[str]) -> float:
    if not values_a or not values_b:
        return 0.0
    best = 0.0
    for value_a in values_a[:12]:
        for value_b in values_b[:12]:
            best = max(best, SequenceMatcher(None, value_a, value_b).ratio())
            if best >= 0.999:
                return best
    return best


def analyze_text_descriptors(query_descriptor: Optional[Dict], target_descriptor: Optional[Dict]) -> Dict:
    if not query_descriptor or not target_descriptor:
        return {
            "fusion_score": 0.0,
            "token_similarity": 0.0,
            "number_similarity": 0.0,
            "identifier_similarity": 0.0,
            "chargram_similarity": 0.0,
            "sequence_similarity": 0.0,
            "line_similarity": 0.0,
            "exact_identifier_match": 0.0,
            "partial_identifier_match": 0.0,
            "quality": 0.0,
        }
    if not query_descriptor.get("has_text") or not target_descriptor.get("has_text"):
        return {
            "fusion_score": 0.0,
            "token_similarity": 0.0,
            "number_similarity": 0.0,
            "identifier_similarity": 0.0,
            "chargram_similarity": 0.0,
            "sequence_similarity": 0.0,
            "line_similarity": 0.0,
            "exact_identifier_match": 0.0,
            "partial_identifier_match": 0.0,
            "quality": 0.0,
        }

    def jaccard(a: List[str], b: List[str]) -> float:
        set_a = set(a or [])
        set_b = set(b or [])
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    token_sim = jaccard(query_descriptor.get("tokens", []), target_descriptor.get("tokens", []))
    number_sim = jaccard(query_descriptor.get("number_tokens", []), target_descriptor.get("number_tokens", []))
    identifier_tokens_a = query_descriptor.get("identifier_tokens", [])
    identifier_tokens_b = target_descriptor.get("identifier_tokens", [])
    identifier_sim = jaccard(identifier_tokens_a, identifier_tokens_b)
    chargram_sim = jaccard(query_descriptor.get("chargrams", []), target_descriptor.get("chargrams", []))
    seq_sim = SequenceMatcher(
        None,
        query_descriptor.get("raw_text", ""),
        target_descriptor.get("raw_text", ""),
    ).ratio()
    line_sim = _max_sequence_similarity(
        query_descriptor.get("text_lines", []),
        target_descriptor.get("text_lines", []),
    )
    partial_identifier_match = _max_sequence_similarity(identifier_tokens_a, identifier_tokens_b)
    exact_identifier_match = 1.0 if set(identifier_tokens_a) & set(identifier_tokens_b) else 0.0
    quality = min(
        float(query_descriptor.get("text_quality", 0.0)),
        float(target_descriptor.get("text_quality", 0.0)),
    )

    score = (
        0.20 * token_sim
        + 0.15 * number_sim
        + 0.30 * identifier_sim
        + 0.15 * chargram_sim
        + 0.10 * seq_sim
        + 0.10 * line_sim
    )
    score = float(max(0.0, min(1.0, score)))
    return {
        "fusion_score": score,
        "token_similarity": float(token_sim),
        "number_similarity": float(number_sim),
        "identifier_similarity": float(identifier_sim),
        "chargram_similarity": float(chargram_sim),
        "sequence_similarity": float(seq_sim),
        "line_similarity": float(line_sim),
        "exact_identifier_match": float(exact_identifier_match),
        "partial_identifier_match": float(partial_identifier_match),
        "quality": float(quality),
    }


def compare_text_descriptors(query_descriptor: Optional[Dict], target_descriptor: Optional[Dict]) -> float:
    return float(analyze_text_descriptors(query_descriptor, target_descriptor).get("fusion_score", 0.0))


def build_filename_descriptor(filename: str) -> Dict:
    normalized = normalize_filename_text(filename)
    descriptor = build_text_descriptor([normalized] if normalized else [])
    descriptor["source"] = "filename"
    descriptor["title_block_lines"] = [normalized] if normalized else []
    descriptor["title_block_entries"] = []
    descriptor["title_block_fields"] = extract_title_block_fields(descriptor["title_block_lines"])
    return descriptor


class OCRService:
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger("ocr_service")
        self.engine = None
        self._initialized = False

    def initialize(self):
        if not self.config.innovation.enable_ocr_text_fusion:
            self._initialized = False
            return
        if RapidOCR is None:
            self.logger.warning("rapidocr_onnxruntime is not installed, OCR text fusion disabled.")
            self._initialized = False
            return
        try:
            self.engine = RapidOCR()
            self._initialized = True
            self.logger.info("OCR service initialized.")
        except Exception as exc:  # pragma: no cover
            self.logger.error(f"Failed to initialize OCR service: {exc}", exc_info=True)
            self._initialized = False

    def _build_text_focus_image(self, image: Image.Image, mask_image: Optional[Image.Image]) -> Image.Image:
        if mask_image is None:
            return image.convert("RGB")

        original = np.asarray(image.convert("RGB"), dtype=np.uint8)
        keep_mask = np.asarray(mask_image.convert("L"), dtype=np.uint8)
        if keep_mask.shape[:2] != original.shape[:2]:
            keep_mask = np.asarray(
                mask_image.convert("L").resize(image.size, Image.Resampling.NEAREST),
                dtype=np.uint8,
            )

        removed_mask = keep_mask < 128
        if removed_mask.mean() < 0.005:
            return image.convert("RGB")

        focused = np.full_like(original, 255)
        focused[removed_mask] = original[removed_mask]
        return Image.fromarray(focused, mode="RGB")

    def _build_title_block_crop(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        left = int(width * 0.55)
        top = int(height * 0.65)
        crop = image.crop((left, top, width, height))
        return crop.convert("RGB")

    def _ocr_entries(self, image: Image.Image, source_name: str = "unknown") -> List[Dict]:
        result, _ = self.engine(np.asarray(image))
        entries: List[Dict] = []
        min_conf = self.config.innovation.ocr_min_confidence
        width, height = image.size
        if not result:
            return entries
        for item in result:
            if len(item) < 3:
                continue
            coords = _extract_box_coords(item[0])
            if coords is None:
                continue
            text = normalize_text(str(item[1]))
            score = float(item[2])
            if not text or score < min_conf:
                continue
            x1, y1, x2, y2 = coords
            entries.append(
                {
                    "text": text,
                    "score": score,
                    "x1": x1 / max(width, 1),
                    "y1": y1 / max(height, 1),
                    "x2": x2 / max(width, 1),
                    "y2": y2 / max(height, 1),
                    "cx": (x1 + x2) / (2.0 * max(width, 1)),
                    "cy": (y1 + y2) / (2.0 * max(height, 1)),
                    "width": (x2 - x1) / max(width, 1),
                    "height": (y2 - y1) / max(height, 1),
                    "source": source_name,
                }
            )
        entries.sort(key=lambda item: (item["y1"], item["x1"]))
        return entries

    def _ocr_lines(self, image: Image.Image) -> List[str]:
        return [entry["text"] for entry in self._ocr_entries(image)]

    def extract_text_descriptor(self, image: Image.Image, mask_image: Optional[Image.Image] = None) -> Dict:
        if not self._initialized or self.engine is None:
            return {
                "raw_text": "",
                "text_lines": [],
                "title_block_lines": [],
                "title_block_entries": [],
                "tokens": [],
                "number_tokens": [],
                "identifier_tokens": [],
                "chargrams": [],
                "has_text": False,
                "source": "disabled",
                "text_quality": 0.0,
                "title_block_fields": extract_title_block_fields([]),
            }

        focus_images = []
        if self.config.innovation.ocr_use_mask_regions:
            focus_images.append(("masked_regions", self._build_text_focus_image(image, mask_image)))
        focus_images.append(("title_block", self._build_title_block_crop(image)))

        merged_lines: List[str] = []
        title_block_lines: List[str] = []
        title_block_entries: List[Dict] = []
        sources = []
        for source_name, focus_image in focus_images:
            entries = self._ocr_entries(focus_image, source_name=source_name)
            lines = [entry["text"] for entry in entries]
            if not lines:
                continue
            sources.append(source_name)
            if source_name == "title_block":
                title_block_lines = list(lines)
                title_block_entries = entries
            for line in lines:
                if line not in merged_lines:
                    merged_lines.append(line)

        descriptor = build_text_descriptor(
            merged_lines,
            max_text_length=self.config.innovation.ocr_max_text_length,
        )
        descriptor["source"] = "+".join(sources) if sources else "none"
        descriptor["title_block_lines"] = title_block_lines[:16]
        descriptor["title_block_entries"] = _serialize_ocr_entries(title_block_entries)
        descriptor["title_block_fields"] = extract_title_block_fields(
            title_block_lines,
            ocr_entries=title_block_entries,
        )
        return descriptor


_ocr_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
        _ocr_service.initialize()
    return _ocr_service


def init_ocr_service():
    global _ocr_service
    _ocr_service = OCRService()
    _ocr_service.initialize()
