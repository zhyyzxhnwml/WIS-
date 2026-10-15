from __future__ import annotations

"""阶段 4：生成映射样本 + alignment 打分。

LLM 模式：给定问题(F/B/S)与一个锚定远域机制，以及同一问题的若干候选兄弟机制，
模型自行决定需要几个来源机制（至少 1 个）：锚定机制必纳（role=主），其余仅在
确实能提供可迁移关系时纳入（role=辅）。每个机制都先列出其源域内部关系图
source_relations，再把其中可迁移的关系映射到目标问题，最后给出针对该问题的
分步解决方法 solution、关键前提 prerequisites、局限 limitations 与验证方法。

alignment 由代码计算，不采信模型自报的数字；对每个机制分别计算再取最小值：

    单机制得分 = 0.6 × 结构一致性 + 0.4 × 系统性
        结构一致性 = 命中 source_relations 的映射条数 / |source_relations|
        系统性     = 命中映射构成的最长关系链长 / 源域最长关系链长

    样本 alignment = min(每个机制得分)

「命中」的判定是：mapping 条目的 (source, relation) 归一化后必须出现在该机制的
source_relations 中；关系链由 source_relations 的 source→target 边确定性推导。
取 min 是为了守门：任何一个机制守不住 alignment ≥ 0.7，整条样本丢弃，避免用
多机制堆数量刷分。

离线模式：模板化映射（结构合法、质量低），用于跑通。
"""

import threading

from .io import as_text, norm_text, stable_id
from .parallel import pmap

_SYNTH_SYSTEM = (
    "你是跨领域功能迁移生成器。给定一个问题的 FBS 描述、一个锚定机制（来自表面无关领域），"
    "以及若干同一问题的候选来源机制（可选用 0..N 个）。请自行判断需要几个来源机制来支撑迁移："
    "锚定机制必须纳入（role=主），其余机制仅在确实能提供可迁移关系时才纳入（role=辅），宁少勿滥。"
    "对每个纳入的机制，先列出该机制所在源域的内部关系图，再把其中可迁移的关系映射到目标问题上。"
    "最后给出针对该问题的具体解决方法 solution（分步、可直接执行，用换行分隔的编号步骤 1. 2. 3. 等）、"
    "关键前提 prerequisites、局限 limitations 与验证方法 validation_plan。"
    "输出严格 JSON：{\"mechanisms\": [{\"source_domain\": 来源领域, \"mechanism\": 机制名, "
    "\"role\": \"主\" 或 \"辅\", "
    "\"source_relations\": [{\"source\": 源域要素, \"target\": 源域要素, \"relation\": 关系}], "
    "\"mapping\": [{\"source\": 源域要素, \"target\": 目标问题要素, \"relation\": 关系, \"condition\": 触发条件}]}], "
    "\"solution\": 分步解决方法, \"prerequisites\": [关键前提], \"limitations\": [局限], \"validation_plan\": 验证方法}。"
    "mechanisms 至少 1 个，且第一条必须是锚定机制（role=主），其余为辅助（role=辅）；"
    "每个机制的 source_relations 至少 3 条，每条 source/target/relation 均非空，描述该机制在源域内的因果结构；"
    "每个机制的 mapping 中每条的 source 与 relation 必须取自该机制 source_relations 里的某一条，不得新造，"
    "且必须写清 condition 适用前提，不得虚构已验证效果。solution 不得为空。只输出 JSON。"
)

MIN_SOURCE_RELATIONS = 3

_RETRY_SUFFIX = (
    "\n\n上一次输出不合规：{reason}。"
    "请严格按规定 JSON 结构重新输出，只输出 JSON，不要任何解释或 markdown 代码块。"
)

# 整条样本级丢弃的分类计数（线程池共用，用锁保护）。逐条 synthesize_one 里累加，
# 由 synthesize() 在批次结束时统一打印，用于观测修复前后的丢弃率。
_DROP_LOCK = threading.Lock()
_DROP_COUNTS = {}
_DROP_ORDER = ("no_json", "anchor_relations", "anchor_mapping", "anchor_condition", "solution", "other")
_DROP_LABELS = {
    "no_json": "无JSON",
    "anchor_relations": "锚定关系不足",
    "anchor_mapping": "锚定无mapping",
    "anchor_condition": "锚定缺condition",
    "solution": "solution为空",
    "other": "其它",
}


def _count_drop(reason):
    with _DROP_LOCK:
        _DROP_COUNTS[reason] = _DROP_COUNTS.get(reason, 0) + 1


def _drop_reset():
    with _DROP_LOCK:
        _DROP_COUNTS.clear()


def _drop_snapshot():
    with _DROP_LOCK:
        return dict(_DROP_COUNTS)


def _report_drops(dropped, total):
    counts = _drop_snapshot()
    parts = [f"{_DROP_LABELS[key]} {counts[key]}" for key in _DROP_ORDER if counts.get(key)]
    classified = sum(counts.values())
    if dropped - classified > 0:
        parts.append(f"其它 {dropped - classified}")
    detail = f"（{' / '.join(parts)}）" if parts else ""
    print(f"[synthesize] 丢弃 {dropped}/{total}{detail}", flush=True)


def _relation_entries(relations):
    """返回 source/target/relation 三者皆非空的关系条目（不去重，按条目数计）。"""
    return [
        item
        for item in relations
        if isinstance(item, dict)
        and as_text(item.get("source"))
        and as_text(item.get("target"))
        and as_text(item.get("relation"))
    ]


def _mapping_entries(mapping):
    """返回 source/target/relation/condition 四者皆非空的映射条目。"""
    return [
        item
        for item in mapping
        if isinstance(item, dict)
        and as_text(item.get("source"))
        and as_text(item.get("target"))
        and as_text(item.get("relation"))
        and as_text(item.get("condition"))
    ]


def _has_mechanisms(data):
    return isinstance(data, dict) and isinstance(data.get("mechanisms"), list) and bool(data["mechanisms"])


def _anchor_defect(data):
    """锚定机制(index 0)不合规时返回 (类别key, 人类可读原因)；合规返回 None。"""
    anchor = data["mechanisms"][0] if _has_mechanisms(data) and isinstance(data["mechanisms"][0], dict) else None
    if anchor is None:
        return "no_json", "没有有效的 mechanisms[0] 锚定机制对象"
    relations = [r for r in (anchor.get("source_relations") or []) if isinstance(r, dict)]
    if len(_relation_entries(relations)) < MIN_SOURCE_RELATIONS:
        return "anchor_relations", f"锚定机制的 source_relations 有效条目不足 {MIN_SOURCE_RELATIONS} 条"
    mapping = [t for t in (anchor.get("mapping") or []) if isinstance(t, dict)]
    if not mapping:
        return "anchor_mapping", "锚定机制没有 mapping 映射条目"
    if len(_mapping_entries(mapping)) != len(mapping):
        return "anchor_condition", "锚定机制的 mapping 条目缺少 condition 触发条件"
    return None


def _aux_valid(item):
    if not isinstance(item, dict):
        return False
    if not as_text(item.get("source_domain")) or not as_text(item.get("mechanism")):
        return False
    relations = [r for r in (item.get("source_relations") or []) if isinstance(r, dict)]
    if not _relation_entries(relations):
        return False
    mapping = [t for t in (item.get("mapping") or []) if isinstance(t, dict)]
    if not mapping:
        return False
    return len(_mapping_entries(mapping)) == len(mapping)


def _offline_completion(problem, mechanism, siblings):
    element = mechanism.get("function") or mechanism["mechanism"]
    relation = "以源域机制实现目标功能"
    target = problem.get("function") or problem["question"]
    return {
        "mechanisms": [
            {
                "source_domain": mechanism["source_domain"],
                "mechanism": mechanism["mechanism"],
                "role": "主",
                "source_relations": [{"source": element, "target": mechanism["mechanism"], "relation": relation}],
                "mapping": [
                    {
                        "source": element,
                        "target": target,
                        "relation": relation,
                        "condition": "需在目标约束下校验可行性与副作用",
                    }
                ],
            }
        ],
        "solution": f"先在目标场景复现「{mechanism['mechanism']}」的关键关系，再逐步替换为目标问题的实现。",
        "prerequisites": ["目标场景可测", "源域关系可迁移"],
        "limitations": ["离线模板生成，未经人工复核"],
        "validation_plan": "在目标场景下对照实验，记录功能达成与副作用。",
    }


def _prompt(problem, mechanism, siblings):
    lines = [
        f"问题：{problem['question']}",
        f"功能：{problem.get('function', '')}",
        f"行为：{problem.get('behavior', '')}",
        f"结构：{problem.get('structure', '')}",
        f"约束：{'；'.join(problem.get('constraints', [])) or '无'}",
        "",
        "锚定机制（必须纳入，role=主，必须给出完整 source_relations 与 mapping）：",
        f"- 来源领域：{mechanism['source_domain']}；机制：{mechanism['mechanism']}；"
        f"机制说明：{mechanism.get('mechanism_description', '')}；功能：{mechanism.get('function', '')}",
    ]
    if siblings:
        lines.append("可选补充领域（可选用 0..N 个；仅纳入确实能提供可迁移关系的领域，宁少勿滥）：")
        for item in siblings:
            lines.append(
                f"- 来源领域：{item['source_domain']}；机制：{item['mechanism']}；"
                f"机制说明：{item.get('mechanism_description', '')}；功能：{item.get('function', '')}"
            )
    else:
        lines.append("可选补充领域：无（本次仅有锚定机制）。")
    lines.append("请自行判断需要几个来源机制（至少 1 个），为每个机制分别给出 source_relations 与 mapping。")
    return "\n".join(lines)


def _relation_keys(relations):
    keys = set()
    for item in relations:
        if not isinstance(item, dict):
            continue
        key = (norm_text(as_text(item.get("source"))), norm_text(as_text(item.get("relation"))))
        if key[0] and key[1]:
            keys.add(key)
    return keys


def _longest_chain(relations, only_keys=None):
    """由 source_relations 的 source→target 边确定性推导最长关系链长（边数）。

    only_keys 非空时只保留 (source, relation) 命中该集合的边，用于计算映射后的关系链。
    拓扑可能含环（模型偶尔造出回边），DFS 时用路径栈跳过回边，保证终止。
    """
    edges = []
    for item in relations:
        if not isinstance(item, dict):
            continue
        source = norm_text(as_text(item.get("source")))
        target = norm_text(as_text(item.get("target")))
        key = (source, norm_text(as_text(item.get("relation"))))
        if source and target and (only_keys is None or key in only_keys):
            edges.append((source, target))
    adjacency = {}
    nodes = set()
    for source, target in edges:
        adjacency.setdefault(source, []).append(target)
        nodes.update((source, target))
    memo = {}

    def dfs(node, path):
        if node in memo:
            return memo[node]
        best = 0
        for nxt in adjacency.get(node, []):
            if nxt in path:
                continue
            best = max(best, 1 + dfs(nxt, path | {nxt}))
        memo[node] = best
        return best

    return max((dfs(node, {node}) for node in nodes), default=0)


def _mechanism_score(mechanism):
    relations = [r for r in (mechanism.get("source_relations") or []) if isinstance(r, dict)]
    keys = _relation_keys(relations)
    mapping = [t for t in (mechanism.get("mapping") or []) if isinstance(t, dict)]
    matched_keys = set()
    matched = 0
    for item in mapping:
        key = (norm_text(as_text(item.get("source"))), norm_text(as_text(item.get("relation"))))
        if key in keys:
            matched += 1
            matched_keys.add(key)
    consistency = min(1.0, matched / max(1, len(keys)))
    source_chain = max(1, _longest_chain(relations))
    mapped_chain = min(source_chain, _longest_chain(relations, matched_keys))
    systematic = mapped_chain / source_chain
    return round(0.6 * consistency + 0.4 * systematic, 4)


def alignment_score(completion):
    mechanisms = [m for m in (completion.get("mechanisms") or []) if isinstance(m, dict)]
    if not mechanisms:
        return 0.0
    return min(_mechanism_score(m) for m in mechanisms)


def _normalize_mechanism(item, index, mechanism):
    relations = [r for r in (item.get("source_relations") or []) if isinstance(r, dict)]
    mapping = [t for t in (item.get("mapping") or []) if isinstance(t, dict)]
    return {
        "source_domain": as_text(item.get("source_domain")) or (mechanism["source_domain"] if index == 0 else ""),
        "mechanism": as_text(item.get("mechanism")) or (mechanism["mechanism"] if index == 0 else ""),
        "role": "主" if index == 0 else "辅",
        "source_relations": [
            {
                "source": as_text(r.get("source")),
                "target": as_text(r.get("target")),
                "relation": as_text(r.get("relation")),
            }
            for r in _relation_entries(relations)
        ],
        "mapping": [
            {
                "source": as_text(t.get("source")),
                "target": as_text(t.get("target")),
                "relation": as_text(t.get("relation")),
                "condition": as_text(t.get("condition")),
            }
            for t in mapping
        ],
    }


def synthesize_one(problem, mechanism, siblings, client, offline):
    if client is not None and not offline:
        user = _prompt(problem, mechanism, siblings)
        data = client.chat_json(_SYNTH_SYSTEM, user, temperature=0.6, max_tokens=2048)
        defect = _anchor_defect(data)
        if defect:
            # 锚定机制不合格（或没解析出 JSON）：把原因讲清楚再问一次，仅重试一次。
            detail = defect[1] if _has_mechanisms(data) else "没有解析出合法的 JSON"
            retry_system = _SYNTH_SYSTEM + _RETRY_SUFFIX.format(reason=detail)
            data = client.chat_json(retry_system, user, temperature=0.6, max_tokens=2048)
            defect = _anchor_defect(data) if _has_mechanisms(data) else ("no_json", "解析失败")
            if defect:
                _count_drop(defect[0])
                return None, None
    else:
        data = _offline_completion(problem, mechanism, siblings)

    # 锚定机制(index 0)必在；辅助机制逐个校验，不合规只丢它自己，绝不作废整条。
    mechanisms = []
    for index, item in enumerate(data["mechanisms"]):
        if index == 0:
            mechanisms.append(_normalize_mechanism(item, index, mechanism))
        elif _aux_valid(item):
            mechanisms.append(_normalize_mechanism(item, index, mechanism))

    if not mechanisms:
        _count_drop("other")
        return None, None

    solution = as_text(data.get("solution"))
    if not solution:
        _count_drop("solution")
        return None, None

    completion = {
        "mechanisms": mechanisms,
        "solution": solution,
        "prerequisites": [as_text(x) for x in (data.get("prerequisites") or []) if as_text(x)],
        "limitations": [as_text(x) for x in (data.get("limitations") or []) if as_text(x)],
        "validation_plan": as_text(data.get("validation_plan")),
    }
    return completion, alignment_score(completion)


def synthesize(pairs, problems_by_id, mechanisms_by_id, client=None, offline=False, workers=1):
    siblings_by_problem = {}
    for pair in pairs:
        siblings_by_problem.setdefault(pair["problem_id"], []).append(pair["raw_id"])

    def one(pair):
        problem = problems_by_id.get(pair["problem_id"])
        mechanism = mechanisms_by_id.get(pair["raw_id"])
        if problem is None or mechanism is None:
            return None
        siblings = [
            mechanisms_by_id[raw_id]
            for raw_id in siblings_by_problem.get(pair["problem_id"], [])
            if raw_id != pair["raw_id"]
            and raw_id in mechanisms_by_id
            and mechanisms_by_id[raw_id]["source_domain"] != mechanism["source_domain"]
        ]
        completion, score = synthesize_one(problem, mechanism, siblings, client, offline)
        if completion is None:
            return None
        return {
            "id": stable_id("sft", pair["pair_id"]),
            "problem_group": problem["problem_group"],
            "mechanism_group": pair["mechanism_group"],
            "problem": {
                "question": problem["question"],
                "function": problem.get("function", ""),
                "behavior": problem.get("behavior", ""),
                "structure": problem.get("structure", ""),
                "constraints": list(problem.get("constraints", [])),
            },
            "completion": completion,
            "alignment": score,
            "distance": pair["distance"],
            "source_domain": pair["source_domain"],
            "pool": pair.get("pool"),
        }

    _drop_reset()
    results = pmap(one, pairs, workers, "synthesize")
    rows = [row for row in results if row]
    _report_drops(len(pairs) - len(rows), len(pairs))
    return rows