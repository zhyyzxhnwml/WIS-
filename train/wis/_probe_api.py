from __future__ import annotations

import json
import urllib.error
import urllib.request

from train.wis.pipeline.llm import LLMClient

client = LLMClient()
print("provider =", client.provider)
print("base_url =", client.base_url)
print("model    =", client.model)
print("available=", client.available())

try:
    req = urllib.request.Request(
        f"{client.base_url}/models",
        headers={"Authorization": f"Bearer {client.api_key}"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    print("models:", [m.get("id") for m in body.get("data", [])][:30])
except urllib.error.HTTPError as exc:
    print(f"models HTTPError {exc.code}: {exc.read().decode('utf-8', 'ignore')[:300]}")
except Exception as exc:
    print(f"models FAIL: {type(exc).__name__}: {exc}")

try:
    out = client.chat("你是助手，只输出 JSON。", '输出严格 JSON：{"ok": true}', temperature=0.0, max_tokens=64)
    print("chat ->", repr(out))
except Exception as exc:
    print(f"chat FAIL: {type(exc).__name__}: {exc}")