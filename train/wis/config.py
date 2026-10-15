from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

_REQUIRED = ("model_name_or_path", "train_file", "output_dir")


def load_config(path):
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    for key in _REQUIRED:
        if not raw.get(key):
            raise ValueError(f"config requires {key}")
    entropy_lambda = raw.get("entropy_lambda", 0.0)
    if entropy_lambda < 0:
        raise ValueError("entropy_lambda must be >= 0")
    cond_dropout = raw.get("cond_dropout", 0.0)
    if not 0.0 <= cond_dropout <= 1.0:
        raise ValueError("cond_dropout must be within [0, 1]")
    return raw