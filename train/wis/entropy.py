from __future__ import annotations

import torch


def _chunk_entropy(logits_chunk, mask_chunk):
    """单块的 -Σ p·log p 之和，只统计 mask 位置。

    用 torch.special.entr(p) 而不是 -(p*logp).sum(-1)，少物化一个 [.., V] 张量。
    """
    probs = torch.softmax(logits_chunk.float(), dim=-1)
    entropy = torch.special.entr(probs).sum(-1)
    return (entropy * mask_chunk).sum()


def masked_entropy(logits, labels, chunk=512):
    """按序列分块计算熵正则项。

    2048 长度 × 151936 词表下，整段 fp32 log_softmax 需额外约 2.5GB 显存，
    在 8GB 卡上训练会 OOM；分块 + 激活重算把峰值压到单块的量级。
    """
    logits = logits[:, :-1, :]
    labels = labels[:, 1:]
    mask = labels != -100
    total = mask.sum()
    if not total.any():
        return logits.sum() * 0.0
    acc = None
    for start in range(0, logits.shape[1], chunk):
        part = logits[:, start:start + chunk, :]
        part_mask = mask[:, start:start + chunk]
        if not part_mask.any():
            continue
        if torch.is_grad_enabled() and part.requires_grad:
            from torch.utils.checkpoint import checkpoint

            term = checkpoint(_chunk_entropy, part, part_mask, use_reentrant=False)
        else:
            term = _chunk_entropy(part, part_mask)
        acc = term if acc is None else acc + term
    return acc / total


def sft_trainer_class():
    from transformers import Trainer

    class EntropySFTTrainer(Trainer):
        def __init__(self, *args, entropy_lambda=0.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.entropy_lambda = entropy_lambda

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
            # 必须把 num_items_in_batch 透传给模型：否则 Trainer 走 loss 不除 grad_accum 的分支
            # （trainer.py:3614），梯度会被放大 grad_accum 倍，日志 loss 也随之虚高。
            if num_items_in_batch is not None and getattr(self, "model_accepts_loss_kwargs", False):
                outputs = model(**inputs, num_items_in_batch=num_items_in_batch)
            else:
                outputs = model(**inputs)
            loss = outputs.loss
            if self.entropy_lambda and model.training:
                loss = loss - self.entropy_lambda * masked_entropy(outputs.logits, inputs["labels"])
            return (loss, outputs) if return_outputs else loss

    return EntropySFTTrainer


def dpo_trainer_class():
    from trl import DPOTrainer

    class EntropyDPOTrainer(DPOTrainer):
        def __init__(self, *args, entropy_lambda=0.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.entropy_lambda = entropy_lambda

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            if return_outputs:
                loss, metadata = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
            else:
                loss = super().compute_loss(model, inputs, return_outputs=False, **kwargs)
                metadata = None
            if self.entropy_lambda and model.training:
                prompt_ids = inputs.get("prompt_input_ids")
                chosen_ids = inputs.get("chosen_input_ids")
                if chosen_ids is not None:
                    if prompt_ids is not None:
                        full = torch.cat([prompt_ids, chosen_ids], dim=1)
                        prompt_len = prompt_ids.shape[1]
                    else:
                        full = chosen_ids
                        prompt_len = 0
                    outputs = model(input_ids=full)
                    labels = full.clone()
                    if prompt_len:
                        labels[:, :prompt_len] = -100
                    chosen_mask = inputs.get("chosen_attention_mask")
                    if chosen_mask is not None:
                        span = labels[:, prompt_len:prompt_len + chosen_mask.shape[1]]
                        span[chosen_mask == 0] = -100
                    loss = loss - self.entropy_lambda * masked_entropy(outputs.logits, labels)
            return (loss, metadata) if return_outputs else loss

    return EntropyDPOTrainer