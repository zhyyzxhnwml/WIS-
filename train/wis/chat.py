"""WIS 交互式推理驱动：直接和本地微调后的模型「对话」看效果。

注意：模型训练目标是 render_completion() 生成的中文标签文本（非 JSON），
所以本脚本不做 JSON 解析，只按训练格式构造 prompt 并流式生成 + 结构化摘要。
"""
from __future__ import annotations

import argparse
import gc
import os
import random
import re
import sys
import time
from pathlib import Path

# 与 train.py 保持一致的导入约定：先放开 weights_only，再先 import torch，
# 避免 Windows 上 transformers/pyarrow 抢先加载导致 msvcp140 DLL 遮蔽。
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import torch  # noqa: E402

from .cfg import cfg_scores, sample_token  # noqa: E402
from .collator import prompt_ids, render_completion  # noqa: E402
from .config import ROOT, load_config  # noqa: E402
from .data import read_jsonl  # noqa: E402
from .train import _attach_lora, _load_model, _load_tokenizer  # noqa: E402

DEFAULT_CONFIGS = {
    "sft": "train/wis/configs/local/sft_real.json",
    "dpo": "train/wis/configs/local/dpo_real.json",
}
# 各阶段训练产出的 adapter 目录（SFT 配置无 init_adapter，DPO 配置指向 SFT，
# 都不能直接给 _attach_lora 用，必须由本脚本按 stage 显式写回 cfg）。
DEFAULT_ADAPTERS = {
    "sft": "train/wis/outputs/local-sft-real/adapter",
    "dpo": "train/wis/outputs/local-dpo-real/adapter",
}
EVAL_FILE = "train/wis/data/real_v2_smoke/wis_sft_eval.jsonl"  # 缺省验证集（本地冒烟）
MODEL_CONTEXT = 4096  # 基座 config 的 sliding_window

_MAPPING_RE = re.compile(
    r"^(?P<src>.+?)\s*→\s*(?P<tgt>.+?)：(?P<rel>.+?)(?:（条件：(?P<cond>.*?)）)?\s*$"
)
_MECHANISM_RE = re.compile(r"^-\s*\[(?P<domain>[^\]]*)\]\s*(?P<name>.*)$")


# ---------------------------------------------------------------- 输出后处理
def _parse_mapping_line(text):
    match = _MAPPING_RE.match(text.strip())
    if not match:
        return None
    return {
        "source": match.group("src").strip(),
        "target": match.group("tgt").strip(),
        "relation": match.group("rel").strip(),
        "condition": (match.group("cond") or "").strip(),
    }


def parse_labeled_output(text):
    """把中文标签文本解析成结构化摘要；无法识别任何字段时返回 None（不抛异常）。

    新格式：思路段里 `- [域] 机制名` 是一级行（一个机制），其后的 `  - src → tgt：…`
    是该机制的映射条目；解法为分步文本；关键前提为 `- …` 列表。
    """
    if not isinstance(text, str) or not text.strip():
        return None
    out = {
        "mechanisms": [], "solution": None, "prerequisites": [],
        "limitations": None, "validation_plan": None,
        "status": None, "reason": None, "questions": [], "unparsed": [],
    }
    saw_key = False
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("思路："):
            section = "mechanism"; saw_key = True
        elif line.startswith("解法："):
            section = "solution"; saw_key = True
            rest = line[len("解法："):].strip()
            if rest:
                out["solution"] = rest
        elif line.startswith("关键前提："):
            section = "prerequisite"; saw_key = True
            rest = line[len("关键前提："):].strip()
            if rest:
                out["prerequisites"].append(rest[1:].strip() if rest.startswith("-") else rest)
        elif line.startswith("局限："):
            out["limitations"] = line[len("局限："):].strip(); saw_key = True; section = None
        elif line.startswith("验证计划："):
            out["validation_plan"] = line[len("验证计划："):].strip(); section = None; saw_key = True
        elif line.startswith("状态："):
            out["status"] = line[len("状态："):].strip(); section = None; saw_key = True
        elif line.startswith("原因："):
            out["reason"] = line[len("原因："):].strip(); section = None; saw_key = True
        elif line.startswith("追问："):
            out["questions"].append(line[len("追问："):].strip()); section = None; saw_key = True
        elif section == "mechanism":
            mechanism = _MECHANISM_RE.match(line)
            if mechanism:
                out["mechanisms"].append({
                    "source_domain": mechanism.group("domain").strip(),
                    "mechanism": mechanism.group("name").strip(),
                    "mapping": [],
                })
            elif line.startswith("-") and out["mechanisms"]:
                entry = _parse_mapping_line(line[1:])
                (out["mechanisms"][-1]["mapping"].append(entry) if entry
                 else out["unparsed"].append(line))
            else:
                out["unparsed"].append(line)
        elif section == "solution":
            out["solution"] = (out["solution"] + "\n" + line) if out["solution"] else line
        elif section == "prerequisite":
            if line.startswith("-"):
                out["prerequisites"].append(line[1:].strip())
            else:
                out["unparsed"].append(line)
        else:
            out["unparsed"].append(line)
    return out if saw_key else None


def print_summary(text):
    parsed = parse_labeled_output(text)
    if parsed is None:
        print("结构化摘要：无法解析（原始输出见上）")
        return
    if parsed["status"] is not None or parsed["reason"] is not None:
        print("结构化摘要（拒答分支）：")
        print(f"  状态：{parsed['status'] or '（缺失）'}")
        print(f"  原因：{parsed['reason'] or '（缺失）'}")
        print(f"  追问：{len(parsed['questions'])} 条")
        for i, question in enumerate(parsed["questions"], 1):
            print(f"    {i}. {question}")
        return
    print("结构化摘要：")
    mechanisms = parsed["mechanisms"]
    print(f"  机制数：{len(mechanisms) or '（缺失）'}")
    for i, mechanism in enumerate(mechanisms, 1):
        print(f"    {i}. [{mechanism['source_domain']}] {mechanism['mechanism']}")
        for j, item in enumerate(mechanism["mapping"], 1):
            print(
                f"       {j}. {item['source']} → {item['target']}：{item['relation']}"
                f"（条件：{item['condition'] or '无'}）"
            )
    if parsed["unparsed"]:
        print(f"  （另有 {len(parsed['unparsed'])} 行无法解析）")
    print(f"  解法：{parsed['solution'] or '（缺失）'}")
    prerequisites = parsed["prerequisites"]
    print(f"  关键前提：{('；'.join(prerequisites)) if prerequisites else '（缺失）'}")
    print(f"  局限：{parsed['limitations'] or '（缺失）'}")
    print(f"  验证计划：{parsed['validation_plan'] or '（缺失）'}")


def _short(value, limit=24):
    value = str(value or "").replace("\n", " ")
    return value if len(value) <= limit else value[: limit - 1] + "…"


# ---------------------------------------------------------------- 流式 CFG 生成
def stream_generate(model, tokenizer, prompt, null_prompt, weight, max_new_tokens,
                    temperature, top_p, device, stream=True, on_delta=None):
    """与 cfg.generate_ids 数学一致，但改为流式：逐 token 解码并打印增量文本。

    past_key_values 无法在一次调用里拿到全部 token，故只能自己维护双分支缓存。
    on_delta：可选回调，收到每个稳定增量片段（供 Web 服务把流式结果转交调用方）；
    为 None 且 stream=True 时行为与原来完全一致，不改变既有参数顺序与默认值。
    """
    use_cond = weight != 0.0
    use_uncond = weight != 1.0
    cond_step = list(prompt)
    uncond_step = list(null_prompt)
    cond_past = None
    uncond_past = None
    cond_len = len(prompt)
    uncond_len = len(null_prompt)
    eos = tokenizer.eos_token_id
    generated = []
    printed = 0
    hold = 6  # 中文常被切成多个 BPE token，刚解码时尾部字符可能未补齐
    with torch.no_grad():
        for _ in range(max_new_tokens):
            log_cond = None
            log_uncond = None
            if use_cond:
                out = model(
                    input_ids=torch.tensor([cond_step], device=device),
                    attention_mask=torch.ones((1, cond_len), dtype=torch.long, device=device),
                    past_key_values=cond_past,
                )
                log_cond = out.logits[:, -1, :]
                cond_past = out.past_key_values
            if use_uncond:
                out = model(
                    input_ids=torch.tensor([uncond_step], device=device),
                    attention_mask=torch.ones((1, uncond_len), dtype=torch.long, device=device),
                    past_key_values=uncond_past,
                )
                log_uncond = out.logits[:, -1, :]
                uncond_past = out.past_key_values
            if use_cond and use_uncond:
                scores = cfg_scores(log_cond, log_uncond, weight)
            elif use_cond:
                scores = torch.log_softmax(log_cond.float(), dim=-1)
            else:
                scores = torch.log_softmax(log_uncond.float(), dim=-1)
            token = int(sample_token(scores, temperature, top_p).item())
            generated.append(token)
            if stream or on_delta is not None:
                # 整段 decode 后与上次比较，只发出稳定区间的增量：回退 hold 个字符再发出，
                # 等后续 token 补齐后重新解码这段，避免残留未补齐的替换字符。
                text = tokenizer.decode(generated, skip_special_tokens=True)
                safe_end = max(printed, len(text) - hold)
                if safe_end > printed:
                    delta = text[printed:safe_end]
                    if stream:
                        sys.stdout.write(delta)
                        sys.stdout.flush()
                    if on_delta is not None:
                        on_delta(delta)
                    printed = safe_end
            if token == eos:
                break
            cond_step = [token]
            uncond_step = [token]
            cond_len += 1
            uncond_len += 1
    if stream or on_delta is not None:
        # 生成结束后补发被回退缓冲的尾部（丢弃未补齐的替换字符）
        remainder = tokenizer.decode(generated, skip_special_tokens=True)[printed:].rstrip("\ufffd")
        if remainder:
            if stream:
                sys.stdout.write(remainder)
                sys.stdout.flush()
            if on_delta is not None:
                on_delta(remainder)
    return generated


def generate_once(session, problem, stream=True, weight=None):
    tokenizer = session.tokenizer
    prompt = prompt_ids(tokenizer, problem)
    if session.max_prompt_length and len(prompt) > session.max_prompt_length:
        print(f"[warn] prompt 长度 {len(prompt)} 超过 max_prompt_length={session.max_prompt_length}，已从左侧截断")
        prompt = prompt[-session.max_prompt_length:]
    max_new = session.max_new_tokens
    if len(prompt) + max_new > MODEL_CONTEXT:
        print(f"[warn] prompt({len(prompt)}) + max_new_tokens({max_new}) 超过基座 sliding_window={MODEL_CONTEXT}，超出部分会被滑窗丢弃")
    null_prompt = prompt_ids(tokenizer, None)
    w = session.w if weight is None else weight
    start = time.time()
    generated = stream_generate(
        session.model, tokenizer, prompt, null_prompt, w, max_new,
        session.temperature, session.top_p, session.device, stream,
    )
    elapsed = time.time() - start
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, len(generated), elapsed


# ---------------------------------------------------------------- problem 辅助
def problem_or_none(problem):
    if not problem:
        return None
    has_text = any(str(problem.get(k) or "").strip() for k in ("question", "function", "behavior", "structure"))
    if has_text or problem.get("constraints"):
        return problem
    return None


def problem_digest(problem):
    if not problem:
        return "（空）"
    parts = []
    if problem.get("question"):
        parts.append(f"问题={_short(problem['question'], 40)}")
    for label, key in (("功能", "function"), ("行为", "behavior"), ("结构", "structure")):
        if problem.get(key):
            parts.append(f"{label}={_short(problem[key], 20)}")
    if problem.get("constraints"):
        parts.append(f"约束={len(problem['constraints'])}条")
    return " ".join(parts) if parts else "（空）"


# ---------------------------------------------------------------- 会话
def _infer_device(model):
    # device_map="auto" 下取输入嵌入所在设备最可靠；next(model.parameters()) 可能命中 meta/cpu。
    try:
        embedding = model.get_input_embeddings()
        if embedding is not None:
            return embedding.weight.device
    except Exception:
        pass
    for param in model.parameters():
        if param.device.type != "meta":
            return param.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _prepare_for_inference(model):
    model.eval()
    # prepare_model_for_kbit_training 会顺手打开梯度检查点，推理无需，关掉以免与 use_cache 冲突。
    disable = getattr(model, "gradient_checkpointing_disable", None)
    if callable(disable):
        try:
            disable()
        except Exception:
            pass
    try:
        model.config.use_cache = True
    except Exception:
        pass


class Session:
    def __init__(self, args):
        self.args = args
        self.cfg = None
        self.tokenizer = None
        self.model = None
        self.device = None
        self.adapter_path = None
        self.w = args.w
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.max_new_tokens = args.max_new_tokens
        self.max_prompt_length = args.max_prompt_length
        self.problem = None

    def _cfg(self):
        stage = self.args.stage
        path = Path(self.args.config) if self.args.config else ROOT / DEFAULT_CONFIGS[stage]
        if not path.is_absolute():
            path = ROOT / path
        cfg = load_config(path)
        model_ref = Path(cfg["model_name_or_path"])
        if not model_ref.is_absolute():
            model_ref = ROOT / model_ref
        if not model_ref.exists():
            print(f"[error] 找不到基座模型目录：{model_ref}")
            print(r"        请在仓库根目录（f:\Innomap-master）运行：python -m train.wis.chat")
            raise SystemExit(1)
        cfg["model_name_or_path"] = str(model_ref)
        return cfg

    def _adapter(self):
        if self.args.adapter:
            path = Path(self.args.adapter)
            if not path.is_absolute():
                path = ROOT / path
        else:
            path = ROOT / DEFAULT_ADAPTERS[self.args.stage]
        if not (path / "adapter_config.json").exists():
            print(f"[error] adapter 不存在：{path}")
            print("        请先完成对应阶段训练，或用 --adapter <路径> 指定。")
            raise SystemExit(1)
        return path

    def load(self):
        self.cfg = self._cfg()
        self.adapter_path = self._adapter()
        self.tokenizer = _load_tokenizer(self.cfg)
        model = _load_model(self.cfg, training=False)
        # init_adapter 必须显式写回正确目录，否则 _attach_lora 会新建随机 LoRA 或加载错阶段 adapter。
        attach_cfg = dict(self.cfg)
        attach_cfg["init_adapter"] = str(self.adapter_path)
        attach_cfg["gradient_checkpointing"] = False
        model = _attach_lora(model, attach_cfg, training=False)
        _prepare_for_inference(model)
        self.model = model
        self.device = _infer_device(model)
        self._print_banner()

    def _print_banner(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 ** 2
            reserved = torch.cuda.memory_reserved() / 1024 ** 2
            vram = f"{allocated:.0f}MiB(已分配)/{reserved:.0f}MiB(预留)"
            device_name = torch.cuda.get_device_name(0)
        else:
            vram = "CPU"
            device_name = "CPU"
        print(f"[chat] model  = {self.cfg['model_name_or_path']}")
        print(f"[chat] adapter= {self.adapter_path}")
        print(f"[chat] stage={self.args.stage}  w={self.w:.3f}  device={self.device}  gpu={device_name}  vram={vram}")
        print("[chat] 说明：这是单轮「功能迁移」生成引擎，不是通用对话模型；每次输入都独立按训练格式构造 prompt。")

    def _free(self):
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reload(self, stage, adapter=None):
        self._free()
        self.args.stage = stage
        self.args.adapter = adapter
        self.load()


# ---------------------------------------------------------------- 生成展示
def run_and_show(session, problem=None):
    problem = problem_or_none(session.problem if problem is None else problem)
    print(f"[chat] 提交 problem：{problem_digest(problem)}")
    text, n_tokens, elapsed = generate_once(session, problem, stream=True)
    print()
    print(f"[chat] 生成 {n_tokens} token，耗时 {elapsed:.1f}s")
    print_summary(text)
    return text, n_tokens, elapsed


def run_sweep(session, weights):
    problem = problem_or_none(session.problem)
    print(f"[sweep] 对当前 problem 跑 {len(weights)} 档 CFG：{weights}（{'（空）' if problem is None else problem_digest(problem)}）")
    for w in weights:
        text, n_tokens, elapsed = generate_once(session, problem, stream=False, weight=w)
        parsed = parse_labeled_output(text)
        if parsed is None:
            print(f"  w={w:<4} 解析失败 | {n_tokens} token | {elapsed:.1f}s")
        elif parsed["status"] is not None:
            print(f"  w={w:<4} 拒答：状态={parsed['status']} 原因={_short(parsed['reason'])} | {n_tokens} token | {elapsed:.1f}s")
        else:
            mechanisms = parsed["mechanisms"]
            mapping_count = sum(len(mechanism["mapping"]) for mechanism in mechanisms)
            names = "/".join(_short(mechanism["mechanism"]) for mechanism in mechanisms) or "?"
            print(
                f"  w={w:<4} 机制={len(mechanisms)} 个（{names}） | 映射={mapping_count} 条 | "
                f"{n_tokens} token | {elapsed:.1f}s"
            )


def _parse_weights(text):
    values = []
    for seg in str(text).split(","):
        seg = seg.strip()
        if seg:
            values.append(float(seg))
    if not values:
        raise ValueError("没有可解析的 w 值")
    return values


# ---------------------------------------------------------------- 命令实现
def eval_path(session):
    """验证集路径跟随当前配置的 eval_file，缺省回退到本地冒烟验证集。"""
    cfg = getattr(session, "cfg", None) or {}
    return ROOT / (cfg.get("eval_file") or EVAL_FILE)


def cmd_demo(session, arg):
    path = eval_path(session)
    rows = read_jsonl(path)
    if not rows:
        print(f"[error] 验证集为空：{path}")
        return
    if arg:
        try:
            index = int(arg)
        except ValueError:
            print(f"非法样本序号：{arg!r}（应为整数，从 1 开始）")
            return
        if not 1 <= index <= len(rows):
            print(f"样本序号超出范围：{index}（有效范围 1..{len(rows)}）")
            return
        row = rows[index - 1]
    else:
        row = random.choice(rows)
    session.problem = row["problem"]
    print(f"[demo] 样本 id={row.get('id')}（验证集共 {len(rows)} 条）")
    run_and_show(session, problem=row["problem"])
    print("参考输出（held-out 标注，供对照）：")
    print(render_completion(row["completion"]))


def cmd_set(session, arg):
    parts = arg.split(maxsplit=1)
    keys = ("f", "b", "s", "c")
    if len(parts) != 2 or parts[0] not in keys:
        print("用法：/set f|b|s|c <文本>（f=功能 b=行为 s=结构 c=约束，约束用 ; 分隔）")
        return
    key, value = parts[0], parts[1]
    if session.problem is None:
        session.problem = {}
    if key == "f":
        session.problem["function"] = value
    elif key == "b":
        session.problem["behavior"] = value
    elif key == "s":
        session.problem["structure"] = value
    else:
        session.problem["constraints"] = [seg.strip() for seg in re.split(r"[;；]", value) if seg.strip()]
    print(f"[chat] 已设置 {key} -> {_short(value, 40)}")


def cmd_problem(session):
    problem = session.problem
    if not problem:
        print("当前 problem：空（直接输入文本或 /ask 可设置【问题】）")
        return
    print("当前 problem：")
    print(f"  question   ：{problem.get('question') or '（空）'}")
    print(f"  function   ：{problem.get('function') or '（空）'}")
    print(f"  behavior   ：{problem.get('behavior') or '（空）'}")
    print(f"  structure  ：{problem.get('structure') or '（空）'}")
    constraints = problem.get("constraints") or []
    print(f"  constraints：{'；'.join(str(item) for item in constraints) if constraints else '（空）'}")


def cmd_adapter(session, arg):
    if arg not in ("sft", "dpo"):
        print("用法：/adapter sft|dpo")
        return
    if session.args.adapter is None and arg == session.args.stage:
        print(f"[chat] 当前已是 {arg} adapter")
        return
    target = ROOT / DEFAULT_ADAPTERS[arg]
    if not (target / "adapter_config.json").exists():
        print(f"[error] {arg} adapter 不存在：{target}（保持当前 adapter）")
        return
    print(f"[chat] 切换到 {arg} adapter，释放旧模型显存…")
    session.reload(arg)


def print_help(session):
    print("命令表（含当前取值）：")
    print("  /help              显示本帮助")
    print(f"  /w <数值>          CFG 引导标量：<1 发散 =1 基准 >1 收敛（当前 w={session.w:.3f}）")
    print(f"  /temp <数值>       采样温度（当前 {session.temperature:.3f}）")
    print(f"  /top_p <数值>      top-p 核采样（当前 {session.top_p:.3f}）")
    print(f"  /max <整数>        最大生成 token 数（当前 {session.max_new_tokens}）")
    print("  /ask <文本>        以纯文本作为【问题】提交生成")
    print("  /set f|b|s|c <文本> 填 功能/行为/结构/约束（约束用 ; 分隔）")
    print("  /problem           打印当前待提交的 problem 结构")
    print("  /demo [n]          取验证集第 n 条（缺省随机）生成，并打印参考输出")
    print("  /sweep [w1,w2,...] 对当前问题跑多档 CFG 对比（缺省 0.6,0.8,1.0,1.2）")
    print("  /reset             清空当前 problem")
    print(f"  /adapter sft|dpo   切换 adapter 并重新加载模型（当前 {session.args.stage}）")
    print("  /quit              退出（也支持 Ctrl+C / EOF）")
    print("  其它非 / 输入       直接作为【问题】提交生成")


def _set_number(session, name, arg, attr, cast, extra=None):
    try:
        value = cast(arg)
    except (TypeError, ValueError):
        print(f"[chat] 非法 {name}：{arg!r}，保持原值 {getattr(session, attr)}")
        return
    if extra and not extra(value):
        print(f"[chat] 非法 {name}：{arg!r}，保持原值 {getattr(session, attr)}")
        return
    setattr(session, attr, value)
    print(f"[chat] {name}={value}")


def handle_command(session, line):
    parts = line.split(maxsplit=1)
    cmd = parts[0]
    arg = parts[1].strip() if len(parts) > 1 else ""
    if cmd in ("/quit", "/exit"):
        return False
    if cmd in ("/help", "/"):
        print_help(session)
    elif cmd == "/w":
        _set_number(session, "w", arg, "w", float)
        print(f"[chat] 引导档：{'发散' if session.w < 1 else ('基准' if session.w == 1 else '收敛')}")
    elif cmd == "/temp":
        _set_number(session, "temperature", arg, "temperature", float, lambda value: value > 0)
    elif cmd == "/top_p":
        _set_number(session, "top_p", arg, "top_p", float, lambda value: 0 < value <= 1)
    elif cmd == "/max":
        _set_number(session, "max_new_tokens", arg, "max_new_tokens", int, lambda value: value > 0)
    elif cmd == "/ask":
        if not arg:
            print("用法：/ask <文本>")
        else:
            session.problem = {"question": arg}
            run_and_show(session)
    elif cmd == "/set":
        cmd_set(session, arg)
    elif cmd == "/problem":
        cmd_problem(session)
    elif cmd == "/demo":
        cmd_demo(session, arg)
    elif cmd == "/sweep":
        try:
            weights = _parse_weights(arg) if arg else [0.6, 0.8, 1.0, 1.2]
        except ValueError as exc:
            print(f"[chat] 非法 sweep 参数：{exc}")
        else:
            run_sweep(session, weights)
    elif cmd == "/reset":
        session.problem = None
        print("[chat] 已清空 problem")
    elif cmd == "/adapter":
        cmd_adapter(session, arg)
    else:
        print(f"未知命令：{cmd}（输入 /help 查看命令表）")
    return True


def repl(session):
    print("输入 /help 查看命令；直接输入文本即作为【问题】提交生成。")
    while True:
        try:
            line = input("wis> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if not handle_command(session, line):
                break
        else:
            session.problem = {"question": line}
            run_and_show(session)


# ---------------------------------------------------------------- 入口
def main(argv=None):
    parser = argparse.ArgumentParser(description="WIS 交互式推理驱动（单轮功能迁移生成引擎）")
    parser.add_argument("--config", default=None, help="配置文件路径，缺省按 stage 自动选择")
    parser.add_argument("--stage", choices=["sft", "dpo"], default="sft", help="使用哪一阶段的 adapter")
    parser.add_argument("--adapter", default=None, help="显式指定 adapter 目录，覆盖默认路径")
    parser.add_argument("--w", type=float, default=1.0, help="CFG 引导标量：<1 发散 =1 基准 >1 收敛")
    parser.add_argument("--max-new-tokens", type=int, default=1600, help="最大生成 token 数")
    # 默认值按实测调过：1.5B 欠拟合基座的 logits 尾部很平，temp/top_p 全放开时
    # 会直接产出中英混杂的崩坏片段；0.7/0.9 下同一样本输出通顺且忠于训练格式。
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="top-p 核采样")
    parser.add_argument("--max-prompt-length", type=int, default=512, help="prompt 最大长度，超出从左侧截断")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--ask", default=None, help="单次提问后退出")
    parser.add_argument("--demo", type=int, default=None, help="用验证集第 n 条样本生成并退出")
    parser.add_argument("--sweep", default=None, help="逗号分隔的 w 列表，跑多档对比后退出")
    args = parser.parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    session = Session(args)
    session.load()

    if args.demo is not None:
        cmd_demo(session, str(args.demo))
    elif args.sweep is not None:
        # sweep 优先于 ask：允许 --ask 提供「同一问题」，再对该问题跑多档 CFG 对比
        if args.ask is not None:
            session.problem = {"question": args.ask}
        try:
            weights = _parse_weights(args.sweep)
        except ValueError as exc:
            print(f"[error] 非法 --sweep 参数：{exc}")
            raise SystemExit(1)
        run_sweep(session, weights)
    elif args.ask is not None:
        session.problem = {"question": args.ask}
        run_and_show(session)
    else:
        repl(session)


if __name__ == "__main__":
    main()