from __future__ import annotations

import json
from collections import Counter
from statistics import mean

from train.wis.pipeline.io import read_jsonl
from train.wis.pipeline.templatize import to_flat

OUT = "train/wis/data/v4flash_probe"
cands = read_jsonl(f"{OUT}/candidates.jsonl")
print(f"candidates={len(cands)}")

align = [c["alignment"] for c in cands]
dist = [c["distance"] for c in cands]
print(f"alignment: min={min(align):.3f} mean={mean(align):.3f} max={max(align):.3f}")
print(f"distance : min={min(dist):.3f} mean={mean(dist):.3f} max={max(dist):.3f}")
print("alignment 分布:", Counter(round(a, 1) for a in align))
print("source_domain 分布:", dict(Counter(c["source_domain"] for c in cands)))

train = read_jsonl(f"{OUT}/wis_sft.jsonl")
ev = read_jsonl(f"{OUT}/wis_sft_eval.jsonl")
train_pg = {r["problem_group"] for r in train}
train_mg = {r["mechanism_group"] for r in train}
print(f"train={len(train)} eval={len(ev)}")
print("问题族隔离:", not ({r['problem_group'] for r in ev} & train_pg))
print("机制族隔离:", not ({r['mechanism_group'] for r in ev} & train_mg))

print("\n================ 样例 1 ================")
row = train[0]
flat = to_flat(row)
print(flat["instruction"])
print("---- output ----")
print(flat["output"])
print("\n---- completion 结构 ----")
print(json.dumps(row["completion"], ensure_ascii=False, indent=2)[:2500])

print("\n================ 样例 2 ================")
row = train[len(train) // 2]
flat = to_flat(row)
print(flat["instruction"])
print("---- output ----")
print(flat["output"])