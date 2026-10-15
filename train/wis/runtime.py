from __future__ import annotations

from .config import ROOT
from .data import assert_disjoint, read_jsonl, validate_rows


def preflight(cfg, kind, allow_demo=False):
    train_rows = read_jsonl(ROOT / cfg["train_file"])
    eval_rows = read_jsonl(ROOT / cfg["eval_file"]) if cfg.get("eval_file") else []
    validate_rows(train_rows, kind, allow_demo=allow_demo)
    if eval_rows:
        validate_rows(eval_rows, kind, allow_demo=allow_demo)
        assert_disjoint({"train": train_rows, "eval": eval_rows})
    if kind == "dpo" and cfg.get("cond_dropout", 0.0) > 0:
        if not all(row.get("null_preference_validated") for row in train_rows):
            raise ValueError("null_preference_validated must be true when preference dropout is enabled")
    print(f"[preflight] {kind}: {len(train_rows)} train / {len(eval_rows)} eval rows OK")
    return {"train": len(train_rows), "eval": len(eval_rows)}