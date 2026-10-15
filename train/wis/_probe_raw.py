from __future__ import annotations

import json
import urllib.request

from train.wis.pipeline import synthesize as S
from train.wis.pipeline.io import read_jsonl
from train.wis.pipeline.llm import LLMClient

OUT = "train/wis/data/v4flash_probe"
problems = {p["problem_id"]: p for p in read_jsonl(f"{OUT}/problems.jsonl")}
invs = {m["raw_id"]: m for m in read_jsonl(f"{OUT}/mechanisms_inventory.jsonl")}
pairs = read_jsonl(f"{OUT}/pairs.jsonl")
client = LLMClient()

pair = pairs[0]
user = S._prompt(problems[pair["problem_id"]], invs[pair["raw_id"]])

for mt in (1024, 8192):
    payload = {
        "model": client.model,
        "messages": [
            {"role": "system", "content": S._SYNTH_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.6,
        "max_tokens": mt,
    }
    req = urllib.request.Request(
        f"{client.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {client.api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    choice = body["choices"][0]
    msg = choice["message"]
    print(f"===== max_tokens={mt} =====")
    print("finish_reason:", choice.get("finish_reason"))
    print("usage:", body.get("usage"))
    print("message keys:", list(msg.keys()))
    print("content len:", len(msg.get("content") or ""))
    print("reasoning len:", len(msg.get("reasoning_content") or ""))
    print("reasoning head:", (msg.get("reasoning_content") or "")[:600])
    print("content head:", (msg.get("content") or "")[:600])
    print()