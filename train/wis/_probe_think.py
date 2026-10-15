from __future__ import annotations

import json
import urllib.error
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

variants = {
    "baseline": {},
    "thinking_false": {"thinking": False},
    "thinking_obj_disabled": {"thinking": {"type": "disabled"}},
    "chat_template_kwargs": {"chat_template_kwargs": {"thinking": False}},
    "enable_thinking_false": {"enable_thinking": False},
}

for name, extra in variants.items():
    payload = {
        "model": client.model,
        "messages": [
            {"role": "system", "content": S._SYNTH_SYSTEM},
            {"role": "user", "content": user},
        ],
        "temperature": 0.6,
        "max_tokens": 2048,
    }
    payload.update(extra)
    req = urllib.request.Request(
        f"{client.base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {client.api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        msg = body["choices"][0]["message"]
        usage = body.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        print(
            f"{name:24s} finish={body['choices'][0].get('finish_reason'):8s} "
            f"content={len(msg.get('content') or ''):5d} reasoning={len(msg.get('reasoning_content') or ''):5d} "
            f"reasoning_tokens={details.get('reasoning_tokens')}"
        )
    except urllib.error.HTTPError as exc:
        print(f"{name:24s} HTTP {exc.code}: {exc.read().decode('utf-8', 'ignore')[:200]}")
    except Exception as exc:
        print(f"{name:24s} FAIL {type(exc).__name__}: {exc}")