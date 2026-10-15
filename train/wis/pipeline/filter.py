from __future__ import annotations

"""阶段 5：四道过滤 + 切分 + 落盘训练文件。

四道门：① alignment ≥ 0.7  ② 表面距离 ≥ 0.6  ③ 去重  ④ 训练/验证按问题族+机制族隔离。
产出：wis_sft.jsonl / wis_sft_eval.jsonl / wis_dpo.jsonl / wis_dpo_eval.jsonl。
"""

from pathlib import Path

from ..data import assert_disjoint, validate_rows
from ..prepare import grouped_split
from .io import text_hash, write_jsonl
from .templatize import templatize

MIN_ALIGNMENT = 0.7
MIN_DISTANCE = 0.6


def apply_gates(samples, min_alignment=MIN_ALIGNMENT, min_distance=MIN_DISTANCE):
    kept = []
    seen = set()
    for sample in samples:
        if (sample.get("alignment") or 0.0) < min_alignment:
            continue
        if sample.get("distance", 0.0) < min_distance:
            continue
        key = (text_hash(sample["problem"]["question"]), sample["mechanism_group"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(sample)
    return kept


def make_rejected(problem):
    """构造「表面像答案、实质站不住」的 rejected，用于 DPO 偏好对。

    与 chosen 同量级长度、同样给出思路/分步解法/关键前提，但实质错误：直接照搬
    源域机制不做迁移、与题设约束正面冲突、且自称已验证却没有可验证路径。这样
    chosen/rejected 的差别落在可用性上，而不是靠长度差被模型一眼看穿。
    """
    function = problem.get("function") or problem["question"]
    question = problem.get("question") or function
    structure = problem.get("structure") or "现有结构"
    constraints = problem.get("constraints") or []
    cons_text = "；".join(str(item) for item in constraints) or "（问题未给出约束）"
    return (
        "思路：\n"
        f"- [通用工程] 把「{question}」相关部件整体做大做厚，用体量冗余覆盖需求\n"
        "- [通用工程] 把别处成熟的现成做法原样照搬，不做跨领域迁移\n"
        "- [通用工程] 依靠更强的整体性能自动满足需求，不逐条对应关系\n"
        "解法：\n"
        f"1. 先把「{structure}」按最大可行尺寸整体加厚放大，抬高整体余量；\n"
        "2. 直接把已知的通用方案复制过来，不分析源域与目标场景的差异；\n"
        "3. 按常规经验一次性给出全部参数，不再做条件校验；\n"
        f"4. 忽略题设约束，宣称方案已经满足「{function}」的全部要求；\n"
        "5. 只报告最好的一次结果，不做对照实验，也不评估副作用与代价；\n"
        "6. 认为整体变大变厚即可覆盖所有边界情况，无需迭代修订。\n"
        "关键前提：\n"
        "- 题设约束可以不作考虑\n"
        "- 只要整体更强就一定能满足功能需求\n"
        "- 源域做法可以直接套用，不需要迁移\n"
        f"局限：本方案直接与题设约束（{cons_text}）相冲突，把源域机制原样照搬而没有迁移，"
        "也没有给出可验证的达成路径，结论属于主观断言。\n"
        "验证计划：该做法在其它场景已被验证有效，本问题可直接采用，无需再测。"
    )


def _split_samples(samples):
    """优先用配对阶段已划好的池（两轴天然隔离）；否则回退到按问题族切分+事后剔除。"""
    if any(sample.get("pool") for sample in samples):
        train = [s for s in samples if s.get("pool") == "train"]
        evaluation = [s for s in samples if s.get("pool") == "val"]
        return train, evaluation
    split = grouped_split(samples)
    train = split.get("train", [])
    evaluation = split.get("val", [])
    train_mechanisms = {row["mechanism_group"] for row in train}
    evaluation = [row for row in evaluation if row["mechanism_group"] not in train_mechanisms]
    eval_mechanisms = {row["mechanism_group"] for row in evaluation}
    train = [row for row in train if row["mechanism_group"] not in eval_mechanisms]
    return train, evaluation


def null_preference_validated(sample):
    """条件被 dropout（换成 null prompt）后偏好方向必须不变。

    chosen 必须每个机制都给出可审计的映射（source_relations + 每条带 condition 的
    mapping）；rejected 是结构化非答案文本，没有可审计的机制映射与触发条件，且
    直接照搬机制/违反约束。这个结构差异与具体问题无关，因此去掉条件后偏好依然
    成立——判据不采信模型自报，只看 chosen 是否具备映射结构、rejected 是否为非答案。
    """
    chosen = sample.get("chosen") or {}
    mechanisms = chosen.get("mechanisms") if isinstance(chosen, dict) else None
    chosen_structured = bool(mechanisms)
    if chosen_structured:
        for mechanism in mechanisms:
            mapping = mechanism.get("mapping") if isinstance(mechanism, dict) else None
            if not mapping or not all(item.get("condition") for item in mapping):
                chosen_structured = False
                break
    # rejected 是结构化非答案文本（make_rejected 生成），不得含任何可审计的触发条件。
    # 判据只看「是否存在带条件的映射」，因此必须显式排除「（条件：」——否则任何一句
    # 非空文本都能过门，这道门就失去审计意义。
    rejected = sample.get("rejected")
    rejected_unstructured = (
        isinstance(rejected, str) and bool(rejected.strip()) and "（条件：" not in rejected
    )
    return bool(chosen_structured and rejected_unstructured)


def _dpo_row(sample, index):
    enriched = dict(sample)
    enriched["chosen"] = sample["completion"]
    enriched["rejected"] = make_rejected(sample["problem"])
    enriched["preference_reason"] = (
        "chosen 给出多机制映射与触发条件并落到分步解法；"
        "rejected 表面结构完整但照搬机制、直接违反约束、自称已验证且无可验证路径"
    )
    enriched["null_preference_validated"] = null_preference_validated(enriched)
    return templatize(enriched, index, "dpo")


def write_training_files(samples, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_samples, eval_samples = _split_samples(samples)

    sft_train = [templatize(s, i, "sft") for i, s in enumerate(train_samples)]
    sft_eval = [templatize(s, i, "sft") for i, s in enumerate(eval_samples)]
    validate_rows(sft_train)
    if sft_eval:
        validate_rows(sft_eval)
    assert_disjoint({"train": sft_train, "eval": sft_eval})
    write_jsonl(out_dir / "wis_sft.jsonl", sft_train)
    write_jsonl(out_dir / "wis_sft_eval.jsonl", sft_eval)

    dpo_train = [_dpo_row(s, i) for i, s in enumerate(train_samples)]
    dpo_eval = [_dpo_row(s, i) for i, s in enumerate(eval_samples)]
    validate_rows(dpo_train, "dpo")
    if dpo_eval:
        validate_rows(dpo_eval, "dpo")
    assert_disjoint({"train": dpo_train, "eval": dpo_eval}, optional_mechanism=True)
    write_jsonl(out_dir / "wis_dpo.jsonl", dpo_train)
    write_jsonl(out_dir / "wis_dpo_eval.jsonl", dpo_eval)
    return {
        "sft_train": len(sft_train),
        "sft_eval": len(sft_eval),
        "dpo_train": len(dpo_train),
        "dpo_eval": len(dpo_eval),
    }