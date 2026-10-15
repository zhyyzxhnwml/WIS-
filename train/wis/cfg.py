from __future__ import annotations


def _torch():
    import torch

    return torch


def cfg_scores(cond_logits, uncond_logits, weight):
    torch = _torch()
    log_cond = torch.log_softmax(cond_logits.float(), dim=-1)
    log_uncond = torch.log_softmax(uncond_logits.float(), dim=-1)
    return log_uncond + weight * (log_cond - log_uncond)


def sample_token(logits, temperature=1.0, top_p=1.0):
    torch = _torch()
    logits = logits.float()
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = probs.cumsum(dim=-1)
    remove = cumulative - probs > top_p
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    probs = torch.softmax(sorted_logits, dim=-1)
    choice = torch.multinomial(probs, num_samples=1)
    return sorted_idx.gather(-1, choice).squeeze(-1)


def generate_ids(model, prompt_ids, null_ids, weight, max_new_tokens, top_p=1.0,
                 temperature=1.0, eos_token_ids=None):
    torch = _torch()
    eos = set(eos_token_ids or [])
    use_cond = weight != 0.0
    use_uncond = weight != 1.0
    cond_step = list(prompt_ids)
    uncond_step = list(null_ids)
    cond_past = None
    uncond_past = None
    cond_len = len(prompt_ids)
    uncond_len = len(null_ids)
    generated = []
    for _ in range(max_new_tokens):
        log_cond = None
        log_uncond = None
        if use_cond:
            out = model(input_ids=torch.tensor([cond_step]),
                        attention_mask=torch.ones((1, cond_len), dtype=torch.long),
                        past_key_values=cond_past)
            log_cond = out.logits[:, -1, :]
            cond_past = out.past_key_values
        if use_uncond:
            out = model(input_ids=torch.tensor([uncond_step]),
                        attention_mask=torch.ones((1, uncond_len), dtype=torch.long),
                        past_key_values=uncond_past)
            log_uncond = out.logits[:, -1, :]
            uncond_past = out.past_key_values
        if use_cond and use_uncond:
            scores = cfg_scores(log_cond, log_uncond, weight)
        elif use_cond:
            scores = torch.log_softmax(log_cond.float(), dim=-1)
        else:
            scores = torch.log_softmax(log_uncond.float(), dim=-1)
        token = sample_token(scores, temperature, top_p).item()
        generated.append(token)
        cond_step = [token]
        uncond_step = [token]
        cond_len += 1
        uncond_len += 1
        if token in eos:
            break
    return generated