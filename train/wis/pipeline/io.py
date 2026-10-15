from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

# 9 个目标领域（与原料 field 对齐）；别名归一到规范名
DOMAIN_CANON = ["材料", "生物", "航空航天", "机械", "电子", "计算机", "能源", "环境", "医学"]

_ALIASES = {
    "电子信息": "电子",
    "算法": "计算机",
    "物理": "材料",
    "化工": "材料",
    "建筑": "机械",
    "math": "计算机",
    "cs": "计算机",
}


def norm_domain(value):
    value = (value or "").strip()
    if value in DOMAIN_CANON:
        return value
    if value in _ALIASES:
        return _ALIASES[value]
    for canon in DOMAIN_CANON:
        if canon in value:
            return canon
    return value or "未知"


def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_id(prefix, *parts):
    digest = hashlib.sha1("\u0001".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}-{digest}"


def norm_text(text):
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def text_hash(text):
    return hashlib.sha1(norm_text(text).encode("utf-8")).hexdigest()


def dedup(rows, key):
    seen = set()
    kept = []
    for row in rows:
        value = key(row)
        if value in seen:
            continue
        seen.add(value)
        kept.append(row)
    return kept


def as_text(value, sep="、"):
    """把 LLM 可能返回的 list/None/标量统一成字符串。"""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return sep.join(str(v).strip() for v in value if str(v).strip())
    return str(value).strip()


def as_list(value):
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value).strip()]