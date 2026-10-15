from __future__ import annotations

"""流水线编排 CLI。

    # 用本地 Ollama 造真实样本（默认：qwen3:8b，关思考）
    python -m train.wis.pipeline.run --limit 500 --out-dir train/wis/data

    # 指定本地模型 / 主机
    python -m train.wis.pipeline.run --model qwen2.5:7b-instruct-q4_K_M --host http://localhost:11434

    # 远程 OpenAI 兼容端点
    python -m train.wis.pipeline.run --provider openai --model gpt-4o-mini

    # 离线冒烟（不调 LLM，只跑通结构）
    python -m train.wis.pipeline.run --offline --limit 300 --out-dir train/wis/data/pipeline_smoke

    # 复用已有机制清单，只重跑后半程
    python -m train.wis.pipeline.run --inventory train/wis/data/mechanisms_inventory.jsonl ...
"""

import argparse
import json
import random
import time
from pathlib import Path

from . import domains
from .extract import extract_mechanisms
from .filter import apply_gates, write_training_files
from .io import DOMAIN_CANON, norm_domain, read_jsonl, stable_id, write_jsonl
from .llm import LLMClient
from .pair import mechanism_group, pair_far_domains
from .problems import build_problems
from .synthesize import synthesize


def load_docs(paths, limit, seed=13, balance=True):
    """按领域分层采样。

    原料文件按领域分块顺序排列（arxiv 前 2 万条全是「材料」），顺序截断会让
    --limit 只吃到单一领域，跨领域配对随即失效。这里按领域做蓄水池采样，
    再轮询取满 limit，保证领域均衡。
    """
    rng = random.Random(seed)
    buckets = {}
    counts = {}
    per = max(1, limit // len(DOMAIN_CANON) + 1) if balance else limit
    for path in paths:
        if not Path(path).exists():
            continue
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                if not balance:
                    bucket = buckets.setdefault("_", [])
                    bucket.append(doc)
                    if len(bucket) >= limit:
                        break
                    continue
                domain = norm_domain(doc.get("field"))
                counts[domain] = counts.get(domain, 0) + 1
                bucket = buckets.setdefault(domain, [])
                if len(bucket) < per:
                    bucket.append(doc)
                else:
                    j = rng.randrange(counts[domain])
                    if j < per:
                        bucket[j] = doc
            if not balance and len(buckets.get("_", [])) >= limit:
                break

    if not balance:
        return buckets.get("_", [])[:limit]

    domains = sorted(buckets)
    for domain in domains:
        rng.shuffle(buckets[domain])
    docs = []
    while len(docs) < limit and any(buckets[d] for d in domains):
        for domain in domains:
            if buckets[domain] and len(docs) < limit:
                docs.append(buckets[domain].pop())
    return docs


def _partition(keys, ratios=(0.8, 0.2), seed=13):
    keys = sorted(keys)
    rng = random.Random(seed)
    rng.shuffle(keys)
    if len(keys) <= 1:
        return set(keys), set()
    n = max(1, min(len(keys) - 1, int(round(len(keys) * ratios[0]))))
    return set(keys[:n]), set(keys[n:])


def build_pairs(problems, inventory, k, cross_domain, seed=13, domain_distance=None, min_distance=0.5):
    """按问题族与机制族分别划池，只在池内配对。

    训练/验证必须两轴隔离。若先全量配对、再事后剔除重叠机制，机制池一小
    （30 条量级）验证集就会被剔空——隔离必须在配对之前完成。
    """
    train_pg, val_pg = _partition({p["problem_group"] for p in problems}, seed=seed)
    train_mg, val_mg = _partition({mechanism_group(m) for m in inventory}, seed=seed + 1)
    pairs = []
    for label, problem_groups, mechanism_groups in (
        ("train", train_pg, train_mg),
        ("val", val_pg, val_mg),
    ):
        if not problem_groups or not mechanism_groups:
            continue
        sub_problems = [p for p in problems if p["problem_group"] in problem_groups]
        sub_mechanisms = [m for m in inventory if mechanism_group(m) in mechanism_groups]
        for pair in pair_far_domains(
            sub_problems,
            sub_mechanisms,
            k=k,
            min_distance=min_distance,
            cross_domain=cross_domain,
            domain_distance=domain_distance,
        ):
            pair["pool"] = label
            pairs.append(pair)
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description="WIS 原料→训练样本 流水线")
    parser.add_argument("--raw", default="raw_docs_arxiv.jsonl,raw_docs_openalex.jsonl")
    parser.add_argument("--seed-problems", default=None, help="已有问题 jsonl（含 question）")
    parser.add_argument("--inventory", default=None, help="复用已有机制清单，跳过抽取")
    parser.add_argument("--out-dir", default="train/wis/data")
    parser.add_argument("--limit", type=int, default=1000, help="原料/问题处理上限")
    parser.add_argument("--pairs-per-problem", type=int, default=6)
    parser.add_argument("--seed", type=int, default=13, help="分层采样随机种子")
    parser.add_argument("--no-balance", action="store_true", help="关闭领域均衡，按文件顺序截断")
    parser.add_argument("--allow-same-domain", action="store_true", help="允许同领域配对（默认强制跨域）")
    parser.add_argument(
        "--min-distance",
        type=float,
        default=0.5,
        help="远域门限。语义嵌入下跨域对实测落在 0.50–0.55，原 0.6 是按已失效的词面度量定的",
    )
    parser.add_argument("--offline", action="store_true", help="不调 LLM，走启发式占位")
    parser.add_argument("--provider", choices=["auto", "ollama", "openai"], default="auto")
    parser.add_argument("--model", default=None, help="覆盖默认模型（如 qwen3:8b）")
    parser.add_argument("--host", default=None, help="Ollama 主机，默认 http://localhost:11434")
    parser.add_argument("--think", action="store_true", help="对带思考的模型开启思维链（慢）")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="并发请求数。默认远程 8、本地 Ollama 1（本地并发会互相抢显存，反而更慢）",
    )
    parser.add_argument(
        "--stage",
        choices=["all", "extract", "problems", "pair", "synthesize", "filter"],
        default="all",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = LLMClient(
        provider=args.provider,
        model=args.model,
        host=args.host,
        think=args.think,
    )
    offline = args.offline or not client.available()
    workers = args.workers
    if workers is None:
        # 远程 API 有并发限流（实测 workers=8 会大面积 429），默认降到 4
        workers = 1 if client.provider == "ollama" else 4
    if offline:
        print("[pipeline] 离线模式（--offline 或后端不可用）")
    else:
        print(f"[pipeline] LLM 模式: {client.describe()}  workers={workers}")

    docs = load_docs(args.raw.split(","), args.limit, seed=args.seed, balance=not args.no_balance) if args.raw else []

    inventory_path = Path(args.inventory) if args.inventory else out_dir / "mechanisms_inventory.jsonl"
    if args.inventory:
        inventory = read_jsonl(inventory_path)
        print(f"[extract] 复用机制清单 {len(inventory)} 条")
    elif args.stage in ("all", "extract"):
        print(f"[extract] 原料 {len(docs)} 篇 → 机制清单")
        inventory = extract_mechanisms(docs, client, offline, workers=workers)
        write_jsonl(inventory_path, inventory)
        print(f"[extract] 机制 {len(inventory)} 条 → {inventory_path}")
    else:
        inventory = read_jsonl(inventory_path)

    if args.stage == "extract":
        return

    seed_rows = read_jsonl(args.seed_problems) if args.seed_problems else []
    if args.stage in ("all", "problems"):
        problems = build_problems(seed_rows, docs, client, offline, limit=args.limit, workers=workers)
        write_jsonl(out_dir / "problems.jsonl", problems)
        print(f"[problems] 问题 {len(problems)} 条 → {out_dir / 'problems.jsonl'}")
    else:
        problems = read_jsonl(out_dir / "problems.jsonl")

    if args.stage in ("all", "pair"):
        raw_paths = args.raw.split(",") if args.raw else []
        matrix = domains.load_or_build(raw_paths, out_dir / "domain_distance.json", seed=args.seed)
        if matrix:
            print("[pair] 领域距离矩阵:\n" + domains.describe(matrix))
        else:
            print("[pair] 嵌入模型不可用，退回词面距离（实测区分度极低）")
        pairs = build_pairs(
            problems,
            inventory,
            k=args.pairs_per_problem,
            cross_domain=not args.allow_same_domain,
            seed=args.seed,
            domain_distance=matrix,
            min_distance=args.min_distance,
        )
        write_jsonl(out_dir / "pairs.jsonl", pairs)
        print(f"[pair] 配对 {len(pairs)} 组 → {out_dir / 'pairs.jsonl'}")
    else:
        pairs = read_jsonl(out_dir / "pairs.jsonl")

    if args.stage in ("all", "synthesize"):
        cand_path = out_dir / "candidates.jsonl"
        existing = read_jsonl(cand_path) if cand_path.exists() else []
        done = {row.get("id") for row in existing}
        todo = [p for p in pairs if stable_id("sft", p["pair_id"]) not in done]
        if existing:
            print(f"[synthesize] 续跑：已有 {len(existing)} 条，待生成 {len(todo)}/{len(pairs)}", flush=True)
        problems_by_id = {p["problem_id"]: p for p in problems}
        mechanisms_by_id = {m["raw_id"]: m for m in inventory}
        # 分块落盘：长跑任务一旦中断，已生成的付费样本不会作废，续跑只补缺口
        chunk = 200
        generated = []
        started = time.time()
        for offset in range(0, len(todo), chunk):
            part = todo[offset:offset + chunk]
            generated.extend(
                synthesize(part, problems_by_id, mechanisms_by_id, client, offline, workers=workers)
            )
            write_jsonl(cand_path, existing + generated)
            rate = (offset + len(part)) / max(1e-6, time.time() - started)
            print(
                f"[synthesize] 进度 {offset + len(part)}/{len(todo)}"
                f"，累计候选 {len(existing) + len(generated)} 条"
                f"，{rate:.2f} 条/秒",
                flush=True,
            )
        print(f"[synthesize] 候选样本 {len(existing) + len(generated)} 条 → {cand_path}", flush=True)
        samples = existing + generated
    else:
        cand_path = out_dir / "candidates.jsonl"
        samples = read_jsonl(cand_path) if cand_path.exists() else []

    if args.stage in ("all", "filter"):
        kept = apply_gates(samples, min_distance=args.min_distance)
        print(f"[filter] {len(samples)} → {len(kept)} 条通过四道门")
        stats = write_training_files(kept, out_dir)
        print(f"[filter] 产出: {stats}")


if __name__ == "__main__":
    main()