from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# torch>=2.6 的 torch.load 默认 weights_only=True，会拒绝加载 checkpoint 里
# 由 numpy 写出的优化器/RNG 状态（numpy.core.multiarray._reconstruct）。
# 这些都是本项目自己产出的 checkpoint，放开限制以支持中断续训。
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from .collator import CondDropoutCollator, PreferenceDropoutCollator
from .config import ROOT, load_config
from .data import read_jsonl
from .entropy import dpo_trainer_class, sft_trainer_class
from .runtime import preflight


class _Flag:
    def __init__(self, value=False):
        self.value = value

    def __call__(self):
        return self.value


def _training_flag_callbacks(flag):
    from transformers import TrainerCallback

    class TrainingFlagCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            flag.value = True

        def on_train_end(self, args, state, control, **kwargs):
            flag.value = False

        def on_evaluate(self, args, state, control, **kwargs):
            flag.value = False

        def on_prediction_step(self, args, state, control, **kwargs):
            flag.value = False

    return TrainingFlagCallback()


def _dtype_kwargs(cfg):
    if cfg.get("load_in_4bit"):
        return {"load_in_4bit": True}
    return {"torch_dtype": "auto"}


def _load_tokenizer(cfg):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_model(cfg, training):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    kwargs = {"trust_remote_code": True}
    if cfg.get("load_in_4bit"):
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name_or_path"], **kwargs)
    if training and cfg.get("gradient_checkpointing"):
        model.config.use_cache = False
    return model


def _attach_lora(model, cfg, training=True):
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    if cfg.get("load_in_4bit"):
        model = prepare_model_for_kbit_training(model)

    init = cfg.get("init_adapter")
    init_path = ROOT / init if init else None
    if init_path and init_path.exists():
        # 二阶段必须从一阶段 adapter 继续，否则 SFT 白跑
        model = PeftModel.from_pretrained(model, str(init_path), is_trainable=training)
        print(f"[lora] 从已有 adapter 继续: {init_path}")
    else:
        if init_path:
            print(f"[lora] init_adapter 不存在，改为新建 LoRA: {init_path}")
        lora = LoraConfig(
            r=int(cfg.get("lora_rank", 32)),
            lora_alpha=int(cfg.get("lora_alpha", 64)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=cfg.get(
                "lora_target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        )
        model = get_peft_model(model, lora)

    if cfg.get("gradient_checkpointing"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    return model


def _training_kwargs(cfg):
    base = dict(
        output_dir=str(ROOT / cfg["output_dir"]),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        num_train_epochs=float(cfg.get("num_train_epochs", 3)),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 200)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        bf16=bool(cfg.get("bf16", True)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", True)),
        optim=cfg.get("optim", "paged_adamw_8bit" if cfg.get("load_in_4bit") else "adamw_torch"),
        report_to=cfg.get("report_to", "none"),
        remove_unused_columns=False,
        seed=int(cfg.get("seed", 42)),
        max_grad_norm=float(cfg.get("max_grad_norm", 1.0)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 0)),
    )
    if cfg.get("deepspeed"):
        base["deepspeed"] = str(ROOT / cfg["deepspeed"])
    if cfg.get("eval_file"):
        base["eval_steps"] = int(cfg.get("eval_steps", 200))
        base["eval_strategy"] = "steps"
    return base


def _training_arguments(cfg):
    from transformers import TrainingArguments

    kwargs = _training_kwargs(cfg)
    try:
        return TrainingArguments(**kwargs)
    except TypeError:
        kwargs.pop("eval_strategy", None)
        kwargs["evaluation_strategy"] = "steps"
        return TrainingArguments(**kwargs)


def _make_trainer(trainer_cls, **kwargs):
    try:
        return trainer_cls(processing_class=kwargs.pop("tokenizer"), **kwargs)
    except TypeError:
        return trainer_cls(**kwargs)


def _dataset(rows):
    from datasets import Dataset

    return Dataset.from_list(rows)


def _resume_ckpt(cfg, output_dir):
    """中断后续训：output_dir 下已有 checkpoint 且尚未产出 adapter 时，自动取最大步数的
    checkpoint 续训；已完成（adapter 已落盘）或无 checkpoint 则从零开始。"""
    out = Path(output_dir)
    if (out / "adapter" / "adapter_config.json").exists():
        return None
    cands = sorted(
        out.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    return str(cands[-1]) if cands else None


def train_sft(cfg, allow_demo=False, max_steps=None):
    import torch  # noqa: F401  — 必须先于 transformers/pyarrow 导入，避免 msvcp140 DLL 遮蔽
    from transformers import TrainerCallback  # noqa: F401

    preflight(cfg, "sft", allow_demo=allow_demo)
    tokenizer = _load_tokenizer(cfg)
    model = _load_model(cfg, training=True)
    model = _attach_lora(model, cfg)

    max_length = int(cfg.get("max_length", 2048))
    flag = _Flag(False)
    collator = CondDropoutCollator(
        tokenizer, max_length, probability=float(cfg.get("cond_dropout", 0.0)), training_check=flag
    )
    train_ds = _dataset(read_jsonl(ROOT / cfg["train_file"]))
    eval_ds = _dataset(read_jsonl(ROOT / cfg["eval_file"])) if cfg.get("eval_file") else None

    args = _training_arguments(cfg)
    if max_steps is not None:
        args.max_steps = int(max_steps)

    Trainer = sft_trainer_class()
    trainer = _make_trainer(
        Trainer,
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        entropy_lambda=float(cfg.get("entropy_lambda", 0.0)),
        tokenizer=tokenizer,
        callbacks=[_training_flag_callbacks(flag)],
    )
    resume = _resume_ckpt(cfg, args.output_dir)
    if resume:
        print(f"[sft] 从 checkpoint 续训: {resume}")
    trainer.train(resume_from_checkpoint=resume)
    out = Path(args.output_dir) / "adapter"
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"[sft] adapter saved to {out}")
    return out


def _render_dpo_rows(rows, tokenizer):
    from .collator import render_completion, render_prompt

    rendered = []
    for row in rows:
        # prompt 必须与 SFT / 推理端同源（render_prompt 走 chat 模板），否则 DPO 会
        # 在一个 SFT 没见过的条件格式上做偏好优化，条件分布会被带偏。
        rendered.append(
            {
                "prompt": render_prompt(tokenizer, row["problem"]),
                "chosen": render_completion(row["chosen"]),
                "rejected": row["rejected"],
            }
        )
    return rendered


def train_dpo(cfg, allow_demo=False, max_steps=None):
    import torch  # noqa: F401  — 必须先于 transformers/pyarrow 导入
    from trl import DPOConfig

    preflight(cfg, "dpo", allow_demo=allow_demo)
    tokenizer = _load_tokenizer(cfg)
    model = _load_model(cfg, training=True)
    model = _attach_lora(model, cfg)

    train_ds = _dataset(_render_dpo_rows(read_jsonl(ROOT / cfg["train_file"]), tokenizer))
    eval_ds = (
        _dataset(_render_dpo_rows(read_jsonl(ROOT / cfg["eval_file"]), tokenizer))
        if cfg.get("eval_file")
        else None
    )

    kwargs = _training_kwargs(cfg)
    kwargs.update(
        beta=float(cfg.get("dpo_beta", 0.1)),
        max_length=int(cfg.get("max_length", 2048)),
        max_prompt_length=int(cfg.get("max_prompt_length", cfg.get("max_length", 2048) // 2)),
    )
    dpo_args = DPOConfig(**kwargs)
    if max_steps is not None:
        dpo_args.max_steps = int(max_steps)

    # 偏好 dropout：按 cond_dropout 概率把 prompt 换成 null，保住 SFT 阶段训出的无条件分支，
    # 否则阶段 3 会把发散侧整个洗掉。null prompt 必须与 DPO 分词方式一致（无 special token）。
    # DPOTrainer 的 concatenated_forward 直接取 prompt/chosen/rejected_attention_mask，
    # 必须用会生成 mask 的 PreferenceCollator（DPODataCollatorWithPadding 只做 padding，不补 mask）。
    from trl.trainer.dpo_trainer import PreferenceCollator

    from .collator import prompt_ids

    flag = _Flag(False)
    base_collator = PreferenceCollator(pad_token_id=tokenizer.pad_token_id)
    # null 与 SFT 的 CondDropoutCollator 同源（render_prompt 走 chat 模板），
    # 保证 DPO 的无条件分支与 SFT 学到的 p(y|∅) 对齐。
    null_ids = prompt_ids(tokenizer, None)
    collator = PreferenceDropoutCollator(
        base_collator, null_ids, probability=float(cfg.get("cond_dropout", 0.0)), training_check=flag
    )

    Trainer = dpo_trainer_class()
    trainer = _make_trainer(
        Trainer,
        model=model,
        args=dpo_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        entropy_lambda=float(cfg.get("entropy_lambda", 0.0)),
        callbacks=[_training_flag_callbacks(flag)],
    )
    resume = _resume_ckpt(cfg, dpo_args.output_dir)
    if resume:
        print(f"[dpo] 从 checkpoint 续训: {resume}")
    trainer.train(resume_from_checkpoint=resume)
    out = Path(dpo_args.output_dir) / "adapter"
    trainer.save_model(str(out))
    tokenizer.save_pretrained(str(out))
    print(f"[dpo] adapter saved to {out}")
    return out


def _stage_of(path, override):
    if override:
        return override
    name = Path(path).stem.lower()
    return "dpo" if "dpo" in name else "sft"


def main(argv=None):
    parser = argparse.ArgumentParser(description="WIS 发散式功能迁移训练入口")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--stage", choices=["sft", "dpo"], default=None, help="训练阶段，默认按配置文件名推断")
    parser.add_argument("--allow-demo", action="store_true", help="允许 demo 样本（仅本地冒烟用）")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖 max_steps，便于快速试跑")
    parser.add_argument("--output-dir", default=None, help="覆盖 output_dir，便于试跑时不覆盖正式产物")
    parser.add_argument("--init-adapter", default=None, help="覆盖 init_adapter（二阶段续训/试跑用）")
    parser.add_argument("--model", default=None, help="覆盖 model_name_or_path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.model:
        cfg["model_name_or_path"] = args.model
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.init_adapter:
        cfg["init_adapter"] = args.init_adapter
    stage = _stage_of(args.config, args.stage)
    print(f"[run] stage={stage} model={cfg['model_name_or_path']} config={Path(args.config).name}")
    if stage == "dpo":
        train_dpo(cfg, allow_demo=args.allow_demo, max_steps=args.max_steps)
    else:
        train_sft(cfg, allow_demo=args.allow_demo, max_steps=args.max_steps)


if __name__ == "__main__":
    main()