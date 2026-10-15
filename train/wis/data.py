from __future__ import annotations

import json
from pathlib import Path

_DEFERRAL_STATUSES = ("needs_clarification", "no_candidate")


def read_jsonl(path):
    path = Path(path)
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_response(completion):
    if not isinstance(completion, dict):
        raise ValueError("response must be a JSON object")
    if "status" in completion:
        status = completion["status"]
        if status not in _DEFERRAL_STATUSES:
            raise ValueError(f"unknown response status: {status}")
        reason = completion.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("response.reason is required")
        questions = completion.get("questions")
        if not isinstance(questions, list):
            raise ValueError("response.questions must be a list")
        if status == "needs_clarification" and not questions:
            raise ValueError("response.questions must be non-empty when clarification is needed")
        return
    mechanisms = completion.get("mechanisms")
    if not isinstance(mechanisms, list) or not mechanisms:
        raise ValueError("response.mechanisms must be a non-empty list")
    for mechanism in mechanisms:
        if not isinstance(mechanism, dict):
            raise ValueError("mechanisms entries must be objects")
        for field in ("source_domain", "mechanism"):
            value = mechanism.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"mechanisms.{field} is required")
        relations = mechanism.get("source_relations")
        if not isinstance(relations, list) or not relations:
            raise ValueError("mechanisms.source_relations must be a non-empty list")
        for relation in relations:
            if not isinstance(relation, dict):
                raise ValueError("source_relations entries must be objects")
            for field in ("source", "target", "relation"):
                value = relation.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"source_relations.{field} is required")
        mapping = mechanism.get("mapping")
        if not isinstance(mapping, list) or not mapping:
            raise ValueError("mechanisms.mapping must be a non-empty list")
        for item in mapping:
            if not isinstance(item, dict):
                raise ValueError("mapping entries must be objects")
            for field in ("source", "target", "relation", "condition"):
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"mapping.{field} is required")
    solution = completion.get("solution")
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError("response.solution is required")
    if not isinstance(completion.get("prerequisites"), list):
        raise ValueError("response.prerequisites must be a list")


def validate_rows(rows, kind="sft", allow_demo=False):
    for row in rows:
        if kind == "prompts":
            problem = row.get("problem")
            if not isinstance(problem, dict) or not str(problem.get("question", "")).strip():
                raise ValueError("prompt rows require problem.question")
            continue
        if not allow_demo:
            if row.get("demo"):
                raise ValueError("demo sample not allowed in production data")
            if not row.get("reviewed"):
                raise ValueError("production rows require reviewed=true")
        if not isinstance(row.get("problem"), dict):
            raise ValueError("row requires problem")
        if kind == "dpo":
            if not isinstance(row.get("chosen"), dict):
                raise ValueError("dpo row requires chosen response")
            rejected = row.get("rejected")
            if not isinstance(rejected, str) or not rejected.strip():
                raise ValueError("dpo row requires rejected text")
            validate_response(row["chosen"])
        else:
            if "completion" not in row:
                raise ValueError("row requires completion")
            validate_response(row["completion"])


def assert_disjoint(splits, optional_mechanism=False):
    seen_problem = {}
    seen_mechanism = {}
    seen_question = {}
    for name, rows in splits.items():
        for row in rows:
            problem_group = row.get("problem_group")
            owner = seen_problem.get(problem_group)
            if owner is not None and owner != name:
                raise ValueError(f"problem_group '{problem_group}' appears in multiple splits")
            seen_problem[problem_group] = name
            if optional_mechanism:
                mechanism_group = row.get("mechanism_group")
                owner = seen_mechanism.get(mechanism_group)
                if owner is not None and owner != name:
                    raise ValueError(f"mechanism_group '{mechanism_group}' appears in multiple splits")
                seen_mechanism[mechanism_group] = name
            question = str(row.get("problem", {}).get("question", "")).strip()
            owner = seen_question.get(question)
            if owner is not None and owner != name:
                raise ValueError(f"question '{question}' appears in multiple splits")
            seen_question[question] = name