"""原料 → 训练样本 流水线。

阶段顺序（每个阶段落一个中间文件，可单独重跑）：

    raw docs (arxiv/openalex jsonl)
        └─ extract.py   → mechanisms_inventory.jsonl   （机制清单）
        └─ problems.py  → problems.jsonl               （问题库，含 F/B/S）
        └─ pair.py      → pairs.jsonl                  （远域配对）
        └─ synthesize.py→ candidates.jsonl             （候选样本，含 alignment）
        └─ filter.py    → wis_sft.jsonl / wis_sft_eval.jsonl / wis_dpo.jsonl ...

`llm.py` 提供 OpenAI 兼容客户端；无 API 时各阶段退回离线启发式模式，
结构仍然合法，便于端到端冒烟。真实质量必须走 LLM 模式。
"""