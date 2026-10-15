from __future__ import annotations

"""直连 hf-mirror 下载基座权重（绕开 huggingface_hub 的 HEAD 元数据校验）。

镜像站会中途断流，因此必须校验最终字节数并对未完成文件做 Range 续传，
不能把 EOF 当作正常结束。

用法：
    python train/wis/_fetch_base.py
    DEST=<dir> python train/wis/_fetch_base.py
"""

import os
import sys
import time
import urllib.error
import urllib.request

REPO = os.environ.get("BASE_REPO", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
BASE = os.environ.get("HF_MIRROR", "https://hf-mirror.com").rstrip("/")
DEST = os.environ.get("DEST", r"f:\Innomap-master\models\DeepSeek-R1-Distill-Qwen-1.5B")

FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors",
]


def _head(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return int(resp.headers.get("content-length") or 0)


def download(name, max_attempts=200):
    url = f"{BASE}/{REPO}/resolve/main/{name}"
    out = os.path.join(DEST, name)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    total = _head(url)
    t0 = time.time()

    for attempt in range(max_attempts):
        have = os.path.getsize(out) if os.path.exists(out) else 0
        if total and have >= total:
            print(f"[ok] {name} {have/1e6:.1f}MB (尝试 {attempt} 次)", flush=True)
            return out
        headers = {"Range": f"bytes={have}-"} if have else {}
        mode = "ab" if have else "wb"
        req = urllib.request.Request(url, headers=headers)
        last = have
        try:
            with urllib.request.urlopen(req, timeout=180) as resp, open(out, mode) as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
                    have += len(chunk)
                    if have - last >= 200 << 20:
                        elapsed = max(1e-6, time.time() - t0)
                        pct = f"{100*have/total:.1f}%" if total else "?"
                        print(
                            f"  {name}: {have/1e6:.0f}/{total/1e6:.0f}MB ({pct}) "
                            f"本次均速 {have/1e6/elapsed:.2f}MB/s",
                            flush=True,
                        )
                        last = have
        except (urllib.error.URLError, OSError) as exc:
            print(f"[retry {attempt}] {name} @ {have/1e6:.0f}MB: {exc}", flush=True)
            time.sleep(2)
            continue

        final = os.path.getsize(out) if os.path.exists(out) else 0
        if total and final < total:
            print(f"[incomplete {attempt}] {name} {final/1e6:.0f}/{total/1e6:.0f}MB，续传", flush=True)
            time.sleep(1)
            continue
        print(f"[done] {name} {final/1e6:.1f}MB {time.time()-t0:.0f}s", flush=True)
        return out

    raise RuntimeError(f"{name} 多次尝试后仍未下载完整")


if __name__ == "__main__":
    print(f"[fetch] repo={REPO} mirror={BASE} dest={DEST}", flush=True)
    for name in FILES:
        try:
            download(name)
        except (urllib.error.URLError, OSError, RuntimeError) as exc:
            print(f"[error] {name}: {exc}", file=sys.stderr, flush=True)
            raise
    print("ALL_DONE", flush=True)