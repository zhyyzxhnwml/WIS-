from __future__ import annotations

import random

NULL_QUESTION = "【问题】未提供"

_THINK_OPEN = "\u003cthink\u003e"
_THINK_CLOSE = "\u003c/think\u003e"


def format_problem(problem):
    parts = [f"【问题】{problem.get('question', '')}"]
    for label, key in (("功能", "function"), ("行为", "behavior"), ("结构", "structure")):
        value = problem.get(key)
        if value:
            parts.append(f"【{label}】{value}")
    constraints = problem.get("constraints") or []
    if constraints:
        parts.append("【约束】" + "；".join(str(item) for item in constraints))
    return "\n".join(parts)


def render_prompt(tokenizer, problem):
    content = NULL_QUESTION if problem is None else format_problem(problem)
    messages = [{"role": "user", "content": content}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if _THINK_OPEN not in text:
        text = text + _THINK_OPEN
    return text + _THINK_CLOSE + "\n\n"


def prompt_ids(tokenizer, problem=None):
    return tokenizer.encode(render_prompt(tokenizer, problem), add_special_tokens=False)


def render_completion(completion):
    if "status" in completion:
        lines = [f"状态：{completion['status']}", f"原因：{completion.get('reason', '')}"]
        for question in completion.get("questions", []):
            lines.append(f"追问：{question}")
        return "\n".join(lines)
    lines = ["思路："]
    for mechanism in completion.get("mechanisms", []):
        lines.append(f"- [{mechanism.get('source_domain', '')}] {mechanism.get('mechanism', '')}")
        for item in mechanism.get("mapping", []):
            lines.append(
                f"  - {item['source']} → {item['target']}：{item['relation']}（条件：{item['condition']}）"
            )
    solution = completion.get("solution")
    if solution:
        lines.append("解法：")
        lines.append(solution)
    prerequisites = completion.get("prerequisites") or []
    if prerequisites:
        lines.append("关键前提：")
        lines.extend(f"- {item}" for item in prerequisites)
    if completion.get("limitations"):
        lines.append("局限：" + "；".join(completion["limitations"]))
    if completion.get("validation_plan"):
        lines.append("验证计划：" + completion["validation_plan"])
    return "\n".join(lines)


def _fit(prefix, body, max_length):
    """把 (prefix, body) 压到 max_length 内：优先保住 assistant 段（prefix 尾部的模板标记），
    超长时从左侧截断问题正文；body 本身超长才截断 body 尾部。"""
    if len(prefix) + len(body) <= max_length:
        return prefix, body
    room = max_length - len(body)
    if room >= 0:
        return (prefix[-room:] if room else []), body
    return [], body[:max_length]


def sample_length(tokenizer, row):
    prefix = prompt_ids(tokenizer, row["problem"])
    body = tokenizer.encode(render_completion(row["completion"]), add_special_tokens=False)
    return len(prefix) + len(body) + 1  # +1 为 eos


def encode_sample(tokenizer, row, max_length):
    prefix = prompt_ids(tokenizer, row["problem"])
    body = tokenizer.encode(render_completion(row["completion"]), add_special_tokens=False)
    body = body + [tokenizer.eos_token_id]
    prefix, body = _fit(prefix, body, max_length)
    labels = [-100] * len(prefix) + body
    return {"input_ids": prefix + body, "labels": labels}


class CondDropoutCollator:
    def __init__(self, tokenizer, max_length, probability=0.0, training_check=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.probability = probability
        self.training_check = training_check or (lambda: True)

    def _prefix(self, problem, drop):
        if drop:
            return prompt_ids(self.tokenizer, None)
        return prompt_ids(self.tokenizer, problem)

    def __call__(self, rows):
        import torch

        use_dropout = self.training_check()
        examples = []
        for row in rows:
            drop = use_dropout and random.random() < self.probability
            prefix = self._prefix(row["problem"], drop)
            body = self.tokenizer.encode(render_completion(row["completion"]), add_special_tokens=False)
            body = body + [self.tokenizer.eos_token_id]
            examples.append(_fit(prefix, body, self.max_length))
        longest = max(len(prefix) + len(body) for prefix, body in examples)
        input_ids, attention_mask, labels = [], [], []
        for prefix, body in examples:
            ids = prefix + body
            lab = [-100] * len(prefix) + body
            pad = longest - len(ids)
            attention_mask.append([1] * len(ids) + [0] * pad)
            input_ids.append(ids + [self.tokenizer.pad_token_id] * pad)
            labels.append(lab + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class PreferenceDropoutCollator:
    def __init__(self, base_collator, null_prompt_ids, probability=0.0, training_check=None):
        self.base_collator = base_collator
        self.null_prompt_ids = list(null_prompt_ids)
        self.probability = probability
        self.training_check = training_check or (lambda: True)

    def __call__(self, rows):
        use_dropout = self.training_check()
        prepared = []
        for row in rows:
            if "ref_chosen_logps" in row or "ref_rejected_logps" in row:
                raise ValueError("preference rows must not carry cached reference logps (caching)")
            updated = dict(row)
            if use_dropout and random.random() < self.probability:
                updated["prompt_input_ids"] = list(self.null_prompt_ids)
            prepared.append(updated)
        return self.base_collator(prepared)