from __future__ import annotations

import json

from train.wis.pipeline import synthesize as S
from train.wis.pipeline.io import read_jsonl
from train.wis.pipeline.llm import LLMClient, _extract_json

OUT = "train/wis/data/v4flash_probe"
problems = {p["problem_id"]: p for p in read_jsonl(f"{OUT}/problems.jsonl")}
invs = {m["raw_id"]: m for m in read_jsonl(f"{OUT}/mechanisms_inventory.jsonl")}
pairs = read_jsonl(f"{OUT}/pairs.jsonl")
client = LLMClient()

print(f"problems={len(problems)} mechanisms={len(invs)} pairs={len(pairs)}")
print("=== 样例问题 ===")
print(json.dumps(next(iter(problems.values())), ensure_ascii=False, indent=2))
print("=== 样例机制 ===")
print(json.dumps(next(iter(invs.values())), ensure_ascii=False, indent=2))

pair = pairs[0]
problem = problems[pair["problem_id"]]
mech = invs[pair["raw_id"]]
user = S._prompt(problem, mech, [])
print("=== PROMPT ===")
print(user)

for mt in (1024, 4096):
    print(f"\n===== max_tokens={mt} =====")
    text = client.chat(S._SYNTH_SYSTEM, user, temperature=0.6, max_tokens=mt)
    print(f"[len={len(text)}]")
    print(text[:4000])
    data = _extract_json(text)
    print("--- parsed ok:", bool(data))
    if data:
        mechs = data.get("mechanisms") or []
        print("mechanisms:", len(mechs))
        for i, m in enumerate(mechs):
            rels = m.get("source_relations") or []
            maps = m.get("mapping") or []
            print(f"  [{i}] domain={m.get('source_domain')} mechanism={m.get('mechanism')} role={m.get('role')}")
            print(f"      source_relations: {len(rels)} unique keys: {len(S._relation_keys(rels))}")
            print(f"      mapping: {len(maps)} conditions empty: {sum(1 for t in maps if not str(t.get('condition', '')).strip())}")
        print("solution empty:", not str(data.get("solution") or "").strip())