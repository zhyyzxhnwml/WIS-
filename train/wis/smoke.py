from __future__ import annotations

"""本地训练冒烟：用随机初始化的极小 Qwen2 模型 + 现场训练的小分词器，
在 GPU 上真实跑通 SFT 循环（条件 dropout / 熵正则 / LoRA / 保存 adapter）。

不需要下载任何权重，用于验证训练管线在本机可运行。

    python -m train.wis.smoke
"""

import json
import shutil
from pathlib import Path

from .config import ROOT
from .data import read_jsonl
from .train import train_sft


def _build_tokenizer(corpus, out_dir):
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    trainer = ByteLevelBPETokenizer()
    trainer.train_from_iterator(corpus, vocab_size=512, min_frequency=1,
                                special_tokens=["<pad>", "<eos>", "<unk>"])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=trainer,
        bos_token="<eos>", eos_token="<eos>", unk_token="<unk>", pad_token="<pad>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}<{{ message['role'] }}>{{ message['content'] }}</{{ message['role'] }}>"
        "{% endfor %}<assistant>"
    )
    tokenizer.save_pretrained(str(out_dir))
    return tokenizer


def _build_model(tokenizer, out_dir):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    model = Qwen2ForCausalLM(config)
    model.save_pretrained(str(out_dir))
    return model


def main():
    import argparse

    import torch

    parser = argparse.ArgumentParser(description="本地训练冒烟")
    parser.add_argument("--stage", choices=["sft", "dpo", "both"], default="sft")
    args = parser.parse_args()

    data_dir = ROOT / "train/wis/data/pipeline_smoke"
    if not (data_dir / "wis_sft.jsonl").exists():
        raise SystemExit("先运行: python -m train.wis.pipeline.run --offline --limit 400 --out-dir train/wis/data/pipeline_smoke")

    model_dir = ROOT / "train/wis/outputs/_smoke_model"
    if model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(data_dir / "wis_sft.jsonl")[:200]
    corpus = [json.dumps(row, ensure_ascii=False) for row in rows]
    tokenizer = _build_tokenizer(corpus, model_dir)
    _build_model(tokenizer, model_dir)
    print(f"[smoke] 极小模型已就绪 vocab={len(tokenizer)} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")

    common = {
        "model_name_or_path": str(model_dir),
        "load_in_4bit": False,
        "bf16": True,
        "max_length": 512,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "num_train_epochs": 1,
        "learning_rate": 0.0001,
        "lora_rank": 8,
        "lora_alpha": 16,
        "gradient_checkpointing": False,
        "logging_steps": 1,
        "save_steps": 100000,
        "eval_steps": 100000,
        "report_to": "none",
        "seed": 42,
    }
    if args.stage in ("sft", "both"):
        cfg = dict(
            common,
            train_file="train/wis/data/pipeline_smoke/wis_sft.jsonl",
            eval_file="train/wis/data/pipeline_smoke/wis_sft_eval.jsonl",
            output_dir="train/wis/outputs/_smoke_run_sft",
            cond_dropout=0.15,
            entropy_lambda=0.005,
        )
        train_sft(cfg, allow_demo=False, max_steps=3)
        print("[smoke] SFT 训练循环跑通")
    if args.stage in ("dpo", "both"):
        from .train import train_dpo

        cfg = dict(
            common,
            train_file="train/wis/data/pipeline_smoke/wis_dpo.jsonl",
            eval_file="train/wis/data/pipeline_smoke/wis_dpo_eval.jsonl",
            output_dir="train/wis/outputs/_smoke_run_dpo",
            learning_rate=5e-6,
            dpo_beta=0.1,
            max_prompt_length=256,
            cond_dropout=0.0,
            entropy_lambda=0.005,
        )
        train_dpo(cfg, allow_demo=False, max_steps=3)
        print("[smoke] DPO 训练循环跑通")


if __name__ == "__main__":
    main()