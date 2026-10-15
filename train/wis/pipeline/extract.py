from __future__ import annotations

"""阶段 1：原料 → 机制清单。

LLM 模式：从标题+摘要抽取「可迁移机制」，规范为 {mechanism, description, instance, function}。
离线模式：启发式占位（标题作机制名、摘要首句作描述），仅用于跑通结构。
"""

from .io import as_text, dedup, norm_domain, norm_text, stable_id, text_hash
from .parallel import pmap

_EXTRACT_SYSTEM = (
    "你是跨领域机制抽取器。从给定文献标题与摘要中，抽取一个可迁移的机制（不绑定具体对象），"
    "输出严格 JSON：{\"mechanism\": 机制名, \"mechanism_description\": 一句话机制说明, "
    "\"instance\": 该机制在本文献中的具体实例, \"function\": 该机制实现的功能}。"
    "只输出 JSON，不要解释。"
)


def _clean(text, limit=240):
    return norm_text(text)[:limit]


def extract_one(doc, client, offline):
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    domain = norm_domain(doc.get("field"))
    if client is not None and not offline:
        user = f"领域：{domain}\n标题：{title}\n摘要：{text[:1500]}"
        data = client.chat_json(_EXTRACT_SYSTEM, user)
        if not data or not str(data.get("mechanism", "")).strip():
            return None
        return {
            "raw_id": stable_id("mech", doc.get("source"), title, text_hash(text)),
            "mechanism": as_text(data["mechanism"]),
            "mechanism_description": as_text(data.get("mechanism_description")),
            "instance": as_text(data.get("instance")) or title,
            "source_domain": domain,
            "source": doc.get("source", ""),
            "url": (doc.get("meta") or {}).get("url") or (doc.get("meta") or {}).get("doi", ""),
            "function": as_text(data.get("function")),
            "reviewed": False,
            "needs": ["target_problem", "relation_mapping", "human_review"],
        }
    if not title:
        return None
    head = title.split("：")[0].split(":")[0].strip()
    return {
        "raw_id": stable_id("mech", doc.get("source"), title, text_hash(text)),
        "mechanism": head[:60],
        "mechanism_description": _clean(text),
        "instance": title[:120],
        "source_domain": domain,
        "source": doc.get("source", ""),
        "url": (doc.get("meta") or {}).get("url") or (doc.get("meta") or {}).get("doi", ""),
        "function": _clean(text, 80),
        "reviewed": False,
        "needs": ["target_problem", "relation_mapping", "human_review"],
    }


def extract_mechanisms(docs, client=None, offline=False, limit=None, workers=1):
    selected = docs[:limit] if limit is not None else list(docs)
    rows = [r for r in pmap(lambda d: extract_one(d, client, offline), selected, workers, "extract") if r]
    rows = dedup(rows, lambda r: (norm_text(r["mechanism"]), r["source_domain"]))
    return rows