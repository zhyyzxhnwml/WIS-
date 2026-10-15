from __future__ import annotations

"""领域距离：用本地 Ollama 嵌入模型度量领域之间的语义距离。

字符 n-gram TF-IDF 在这份语料上不可用——原料 99.8% 是英文学术摘要，字符 n-gram
被高频字母对主导，实测各领域质心距离全在 0.018 以内，远域配对彻底失去区分度。
改用 nomic-embed-text 的语义向量后，不同领域文本距离约 0.63，「距离 ≥ 0.6」这道门
才真正起作用。
"""

import json
import random
import urllib.error
import urllib.request
from pathlib import Path

from .io import DOMAIN_CANON, norm_domain

DEFAULT_EMBED_HOST = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


def _sample_docs(paths, per_domain, seed):
    rng = random.Random(seed)
    buckets = {}
    counts = {}
    for path in paths:
        if not Path(path).exists():
            continue
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                domain = norm_domain(doc.get("field"))
                text = f"{doc.get('title') or ''}. {(doc.get('text') or '')[:600]}"
                counts[domain] = counts.get(domain, 0) + 1
                bucket = buckets.setdefault(domain, [])
                if len(bucket) < per_domain:
                    bucket.append(text)
                else:
                    j = rng.randrange(counts[domain])
                    if j < per_domain:
                        bucket[j] = text
    return buckets


def _embed(host, model, texts, timeout=300):
    payload = json.dumps({"model": model, "input": texts}).encode("utf-8")
    request = urllib.request.Request(
        f"{host}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return body.get("embeddings") or []


def available(host=DEFAULT_EMBED_HOST, model=DEFAULT_EMBED_MODEL):
    try:
        request = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(request, timeout=5) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return False
    names = {m.get("name", "") for m in body.get("models", [])}
    return any(name == model or name.startswith(f"{model}:") for name in names)


def build_matrix(paths, per_domain=120, seed=17, host=DEFAULT_EMBED_HOST, model=DEFAULT_EMBED_MODEL, batch=32):
    """域间距离用平均链接：两域文档两两余弦相似度的均值。

    用质心会抹平方差——120 篇求平均后各领域质心几乎重合，实测最大距离只有 0.146，
    过不了 0.6 门。逐对求均值保留绝对尺度（单篇之间相似度约 0.37，距离约 0.63）。
    """
    import numpy as np

    buckets = _sample_docs(paths, per_domain, seed)
    domains = [d for d in DOMAIN_CANON if buckets.get(d)]

    embeddings = {}
    for domain in domains:
        texts = buckets[domain]
        vectors = []
        for start in range(0, len(texts), batch):
            vectors.extend(_embed(host, model, texts[start : start + batch]))
        array = np.asarray(vectors, dtype=float)
        if array.size == 0:
            continue
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings[domain] = array / norms

    distances = {}
    for a in embeddings:
        for b in embeddings:
            if a == b:
                distances[f"{a}|{b}"] = 0.0
                continue
            similarity = float(np.clip((embeddings[a] @ embeddings[b].T).mean(), 0.0, 1.0))
            distances[f"{a}|{b}"] = round(1.0 - similarity, 4)
    return {"domains": sorted(embeddings), "model": model, "matrix": distances}


def load_or_build(paths, cache_path, per_domain=120, seed=17, host=DEFAULT_EMBED_HOST, model=DEFAULT_EMBED_MODEL):
    cache = Path(cache_path)
    if cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data.get("matrix"):
                return data["matrix"]
        except (json.JSONDecodeError, OSError):
            pass
    if not available(host, model):
        return None
    data = build_matrix(paths, per_domain=per_domain, seed=seed, host=host, model=model)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data["matrix"]


def describe(matrix):
    domains = sorted({key.split("|")[0] for key in matrix})
    lines = ["      " + "".join(f"{d:>8s}" for d in domains)]
    for a in domains:
        cells = "".join(f"{matrix.get(f'{a}|{b}', 0.0):8.3f}" for b in domains)
        lines.append(f"{a:>6s}{cells}")
    return "\n".join(lines)