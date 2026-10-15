from __future__ import annotations

import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", r"F:\modelcache\huggingface")

from huggingface_hub import snapshot_download  # noqa: E402

REPO = os.environ.get("BASE_REPO", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")

if __name__ == "__main__":
    print(f"[download] repo={REPO} endpoint={os.environ['HF_ENDPOINT']} home={os.environ['HF_HOME']}", flush=True)
    path = snapshot_download(REPO, max_workers=8)
    print(f"SNAPSHOT:{path}", flush=True)