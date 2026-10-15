from __future__ import annotations

import json
import urllib.request

import numpy as np

HOST = "http://localhost:11434"
MODEL = "nomic-embed-text"

texts = [
    "molecular sieve membrane for hydrogen separation via solvothermal crystallization",
    "deep learning pathology image retrieval digital atlas breast tumor classification",
    "wind turbine blade fatigue crack monitoring using acoustic emission sensors",
]


def embed(batch):
    payload = json.dumps({"model": MODEL, "input": batch}).encode("utf-8")
    req = urllib.request.Request(
        f"{HOST}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    print("response keys:", list(body.keys()))
    return body["embeddings"]


vecs = embed(texts)
arr = np.array(vecs)
print("shape:", arr.shape)
norm = arr / np.linalg.norm(arr, axis=1, keepdims=True)
sim = norm @ norm.T
print("cosine similarity matrix:")
print(np.round(sim, 4))
print("distance (1-sim):")
print(np.round(1 - sim, 4))