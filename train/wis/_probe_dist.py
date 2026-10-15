from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from train.wis.pipeline.io import read_jsonl
from train.wis.pipeline.pair import _mechanism_text, _problem_text

OUT = "train/wis/data/v4flash_probe"
problems = read_jsonl(f"{OUT}/problems.jsonl")
invs = read_jsonl(f"{OUT}/mechanisms_inventory.jsonl")
ptexts = [_problem_text(p) for p in problems]
mtexts = [_mechanism_text(m) for m in invs]

CONFIGS = [
    ("char_wb (2,4) 现用", dict(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
    ("char_wb (1,2)", dict(analyzer="char_wb", ngram_range=(1, 2), min_df=1)),
    ("char (1,2)", dict(analyzer="char", ngram_range=(1, 2), min_df=1)),
    ("char (1,3) sublinear", dict(analyzer="char", ngram_range=(1, 3), min_df=1, sublinear_tf=True)),
]

for name, kw in CONFIGS:
    vec = TfidfVectorizer(**kw)
    mat = vec.fit_transform(ptexts + mtexts)
    pmat = mat[: len(ptexts)]
    mmat = mat[len(ptexts):]
    pn = np.sqrt(pmat.multiply(pmat).sum(axis=1)).A1
    mn = np.sqrt(mmat.multiply(mmat).sum(axis=1)).A1
    pn[pn == 0] = 1.0
    mn[mn == 0] = 1.0
    sim = (pmat @ mmat.T).toarray() / np.outer(pn, mn)
    dist = 1.0 - np.clip(sim, 0, 1)
    print(
        f"{name:22s} dim={mat.shape[1]:6d} dist min={dist.min():.3f} "
        f"p25={np.percentile(dist,25):.3f} median={np.median(dist):.3f} "
        f"max={dist.max():.3f} 跨域对={dist.max()-dist.min():.3f}"
    )