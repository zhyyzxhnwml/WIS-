from __future__ import annotations

"""阶段 2：问题库（含 F/B/S 规范化）。

来源一：已有的问题 jsonl（如人工问题），直接规范化。
来源二：从文献摘要/未来工作提炼科研问题（LLM 模式）。
离线模式：以文献标题模板出占位问题，仅用于跑通结构。
"""

from .io import as_list, as_text, norm_domain, stable_id, text_hash
from .parallel import pmap

_PROBLEM_SYSTEM = (
    "你是问题规范化器。把给定材料转成一条工程/科研问题，并拆成 FBS 三层，输出严格 JSON："
    "{\"question\": 一句话问题, \"function\": 功能(要达到什么), \"behavior\": 行为(过程如何发生), "
    "\"structure\": 结构(涉及哪些部件/载体，写成字符串), \"constraints\": [约束列表,可为空]}。只输出 JSON。"
)


def normalize_problem(row, index):
    question = as_text(row.get("question"))
    return {
        "problem_id": row.get("id") or stable_id("prob", question, index),
        "problem_group": row.get("problem_group") or stable_id("pg", question, index),
        "question": question,
        "function": as_text(row.get("function")),
        "behavior": as_text(row.get("behavior")),
        "structure": as_text(row.get("structure")),
        "constraints": as_list(row.get("constraints")),
        "source_domain": norm_domain(row.get("source_domain") or row.get("field")),
    }


def problem_from_doc(doc, client, offline):
    title = (doc.get("title") or "").strip()
    text = (doc.get("text") or "").strip()
    domain = norm_domain(doc.get("field"))
    if client is not None and not offline:
        user = f"领域：{domain}\n标题：{title}\n摘要：{text[:1500]}"
        data = client.chat_json(_PROBLEM_SYSTEM, user)
        if not data or not as_text(data.get("question")):
            return None
        question = as_text(data["question"])
        return {
            "problem_id": stable_id("prob", title, text_hash(text)),
            "problem_group": stable_id("pg", title, text_hash(text)),
            "question": question,
            "function": as_text(data.get("function")),
            "behavior": as_text(data.get("behavior")),
            "structure": as_text(data.get("structure")),
            "constraints": as_list(data.get("constraints")),
            "source_domain": domain,
        }
    if not title:
        return None
    return {
        "problem_id": stable_id("prob", title, text_hash(text)),
        "problem_group": stable_id("pg", title, text_hash(text)),
        "question": f"如何实现：{title[:80]}",
        "function": "",
        "behavior": "",
        "structure": "",
        "constraints": [],
        "source_domain": domain,
    }


def build_problems(seed_rows, docs, client=None, offline=False, limit=None, workers=1):
    problems = [normalize_problem(row, i) for i, row in enumerate(seed_rows or [])]
    if limit is not None and len(problems) >= limit:
        return problems[:limit]
    seen = {p["question"] for p in problems}
    candidates = pmap(lambda d: problem_from_doc(d, client, offline), list(docs or []), workers, "problems")
    for problem in candidates:
        if limit is not None and len(problems) >= limit:
            break
        if problem and problem["question"] not in seen:
            seen.add(problem["question"])
            problems.append(problem)
    return problems