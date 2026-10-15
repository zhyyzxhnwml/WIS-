from __future__ import annotations

"""确定性模板转换：原始嵌套标注 ↔ 训练文件。

训练文件必须与 `train/wis/collator.py` 消费的嵌套结构一致：
    {id, demo, reviewed, problem_group, mechanism_group, problem{...}, completion{...}}

`to_flat` 额外给出可读的 instruction/output 文本形态，仅用于审计/抽查，
不参与训练（训练文本由 collator 内的 render_prompt/render_completion 生成）。
"""

REVIEWED_REQUIRED = ("mechanisms", "solution")


def templatize(sample, index, kind="sft"):
    problem = sample["problem"]
    row = {
        "id": sample.get("id") or f"{kind}-{index:06d}",
        "demo": False,
        "reviewed": True,
        "problem_group": sample["problem_group"],
        "mechanism_group": sample["mechanism_group"],
        "problem": {
            "question": problem["question"],
            "function": problem.get("function", ""),
            "behavior": problem.get("behavior", ""),
            "structure": problem.get("structure", ""),
            "constraints": list(problem.get("constraints", [])),
        },
    }
    if kind == "dpo":
        row["null_preference_validated"] = bool(sample.get("null_preference_validated", False))
        row["chosen"] = sample.get("chosen") or sample["completion"]
        row["rejected"] = sample["rejected"]
        row["preference_reason"] = sample.get("preference_reason", "")
    else:
        row["completion"] = sample["completion"]
    return row


def to_flat(row):
    problem = row["problem"]
    instr = f"【问题】{problem['question']}\n"
    for label, key in (("功能", "function"), ("行为", "behavior"), ("结构", "structure")):
        if problem.get(key):
            instr += f"【{label}】{problem[key]}\n"
    instr += "【要求】请从一个表面无关的领域给出可迁移机制。"
    completion = row.get("completion") or row.get("chosen") or {}
    if "status" in completion:
        outp = f"状态：{completion['status']}\n原因：{completion.get('reason', '')}"
    else:
        lines = ["思路："]
        for mechanism in completion.get("mechanisms", []):
            lines.append(f"- [{mechanism.get('source_domain', '')}] {mechanism.get('mechanism', '')}")
            for item in mechanism.get("mapping", []):
                lines.append(
                    f"  - {item['source']} → {item['target']}：{item['relation']}（条件：{item['condition']}）"
                )
        if completion.get("solution"):
            lines.append("解法：")
            lines.append(completion["solution"])
        if completion.get("prerequisites"):
            lines.append("关键前提：")
            lines.extend(f"- {item}" for item in completion["prerequisites"])
        if completion.get("limitations"):
            lines.append("局限：" + "；".join(completion["limitations"]))
        if completion.get("validation_plan"):
            lines.append("验证计划：" + completion["validation_plan"])
        outp = "\n".join(lines)
    return {"instruction": instr, "input": "", "output": outp}