# WIS 发散式功能迁移模型 · 训练总结

> 建立在 DeepSeek 之上，用 LoRA 融合训练出的「发散式跨领域思路扩展」模型。它不回答问题，它把问题的**思路重定向**到另一个领域的机制上。本文件覆盖完整训练方案的**理论落点、数据、训练、推理、验收**的工程事实。

---

## 1. 一句话定义

给它一个问题（如「围棋定式选择时的取舍」），它输出的是**从另一个表面无关的领域搬过来的机制**（如「多臂老虎机的探索-利用权衡」），并给出映射关系。

核心认知：**领域是模型的输出，不是你的输入。** 你只抛问题，模型自己钻进所有领域里挑哪个能解决问题。

---

## 2. 三大理论落到训练的三个正交维度

三者不是并列的"指导原则"，而是分别约束三个互不重叠的工程环节：

| 理论 | 提供的机制 | 工程落点 |
|---|---|---|
| **FBS**（Gero） | 功能 F / 行为 B / 结构 S 三层规范 | **输入编码**：把任意问题规范为 F/B/S，在关系层而非表属性层建立可比对形式 |
| **SMT**（Gentner） | 结构对齐、关系优先、系统性 | **对齐度量与损失目标**：`alignment = 0.6×结构一致性 + 0.4×系统性`，SFT 直接拟合该目标 |
| **C-K**（Hatchuel & Weil） | 概念空间 C 与知识空间 K 的四个操作符 | **发散/收敛动力学**：双分支与 CFG 旋钮的抽象来源 |

C-K 的映射是最本质的一层：

- **无条件分支 p(机制|∅)** ≡ **K→C 析取**：从机制知识库无约束析取概念候选（发散）。
- **条件分支 p(机制|问题)** ≡ **C→K 合取**：将概念对接到问题约束下并赋予真值（收敛）。
- **CFG 的 w** ≡ 连续调节「析取 vs 合取」的混合度：w<1 偏向 K→C，w>1 偏向 C→K。
- **末端硬约束判断** ≡ C→K 合取验证的显式化。

**结论**：FBS 决定数据怎么表示，SMT 决定对齐怎么打分，C-K 决定发散-收敛怎么切换。三者映射到「输入 → 目标 → 生成」三段。

---

## 3. 这是什么训练

**一次参数高效指令微调（LoRA-PEFT）+ 直接偏好优化（DPO）的复合训练，独有性质是：通过条件 dropout 联合训练条件分布 p(y|c) 与无条件分布 p(y|∅)，推理时用 CFG 标量 w 在两者间连续插值，得到发散度可调的生成模型。**

标准表述：**CFG-conditioned LoRA instruction tuning + DPO diversity regularization**。

### 3.1 用经典 ML/DL 概念映射

本方案没有任何超出标准框架的新东西，全部可归约到已学概念：

| 经典概念 | 对应 | 公式/形式 |
|---|---|---|
| 监督学习（交叉熵） | SFT | `L = −Σ_t log p_θ(y_t | y_<t, x)` |
| 迁移学习「冻结 backbone 只训头」 | LoRA/QLoRA | `W' = W + A·B, r=32`，只训低秩增量 ΔW |
| dropout（随机丢弃） | 条件 dropout | 以 p 替换条件段，学 p(y|c) 与 p(y|∅) |
| 正则化（L2 惩罚） | 熵正则 | `L_total = L_CE − λ_H·H(p)`，惩罚输出尖峰 |
| RLHF（reward + PPO） | DPO | Bradley-Terry 二元交叉熵，闭式解 |
| 条件概率 / 贝叶斯 | CFG 推理 | log p_γ = log p(y|∅) + γ[log p(y|c) − log p(y|∅)] |

---

## 4. 三个可测量目标

「更发散」是形容词，必须拆成能测的三条，且**由同一份权重同时给出**：

| # | 目标 | 定义 | 阶段 |
|---|---|---|---|
| **T1** | 迁移能力 | 结构对齐 ≥ 0.7 | 阶段 1 |
| **T2** | 发散能力 | 领域覆盖熵显著高于基座；表面距离 ≥ 0.6 | 阶段 2 |
| **T3** | 可切换 | 存在 w 使领域覆盖熵随 w 单调变化 | 2 产出、3 保住 |

「发散 + 收敛」必须融合训练而非「发散模块 + 后置过滤器」，否则末端的收敛会抹平发散。

---

## 5. 基座模型：DeepSeek

选型五原则：① 开源权重（要改权重）② 中文强 ③ 推理强（逻辑是发散的地基）④ 不被 RL 压扁（推理强会熵塌陷）⑤ 单卡可训。

| 算力 | 选择 |
|---|---|
| 1× 4090 | R1-Distill-Qwen-7B / 14B，QLoRA 4bit |
| **2×5090** | **R1-Distill-Qwen-32B（首选）**，QLoRA 4bit（免 offload） |
| 8 卡起 | R1-Distill-Llama-70B |

**风险**：R1-Distill 是蒸馏产物，低熵倾向比原版更重 → 熵正则从「保险」升级为「必需」，`λ_H` 从 0.005 起调。

**工程坑**：模板名必须填 `deepseekr1`（不是 qwen）；` thinking` 段全保留或全剥离，不能一半一半。

---

## 6. 设备需求

**唯一硬门槛是显存（GPU VRAM）。** 内存 64GB 起（造数据）、多核 CPU（算语义距离）、1–2TB NVMe（权重+数据+checkpoint）。

⚠️ **Windows**：DeepSpeed 对原生 Windows 支持差，走 WSL2 或租 Linux 云 GPU。

**关键**：瓶颈在数据不在显卡——数据没造好，给 8×H100 也没用。

---

## 7. 数据

### 7.1 领域覆盖：盯「机制熵」不盯「领域数」

**领域只是标签（桶），机制才是内容（水）。** 发散度上限 = 机制分布的熵，不是领域标签计数。

- 20 个近亲领域 < 5 个彼此极远的领域（前者机制重合）。
- 领域数应**数据驱动**：机制 embed → HDBSCAN 聚类看天然簇数，而非先验拍一个 12 或 15。
- 判断「够不够」用**熵饱和检验**：继续加领域/机制，领域覆盖熵不再上升即达饱和；或用「留出异质领域」测模型能否摸到边界。
- 优先级：**每域机制密度 > 领域异质性 > 领域数量**。一个每域 2000 机制的 9 领域集合，好过每域 200 的 25 领域集合。

### 7.2 数据获取：公开 API 批量拉 + LLM 合成 + 过滤

**几万条不可能人工标注，靠程序化生产。** 三条腿：

**(a) 原料批量拉取（免费公开源）**：

| 原料 | 来源 |
|---|---|
| 学术机制 | arXiv API、Semantic Scholar API、S2ORC、OpenAlex |
| 生物医学机制 | PubMed/PMC E-utilities |
| 工程/算法机制 | Google Patents 公开数据集、USPTO Bulk Data、PatentsView |
| 百科机制 | Wikipedia/Wikidata dump |
| 真实科研问题 | Stack Exchange dump、NSFC 公开摘要、论文 future work/limitations |
| 中文语料 | Aminer 开放图谱（CNKI 有版权限制，不硬爬） |

**(b) LLM 合成（规模靠配对相乘）**：

```
15 领域 × 200 机制 = 3000 机制
5000–10000 问题 × 每问题配 6~8 远域机制 = 3～4 万样本
```

步骤：机制抽取 → 问题规范化（F/B/S）→ embedding+faiss 远域配对 → LLM 生成 p-q-rel 映射。范式 = Self-Instruct / Evol-Instruct。

**(c) 四道过滤（保留率 < 一半，占 60% 工时）**：

1. 双模型交叉验证（不同家族，两轮一致才留）
2. alignment 打分 < 0.7 丢弃
3. embedding 距离 < 0.6 的近域丢弃
4. MinHash / 语义 dedup

**要 3–5 万合格，需拉 8–10 万原料、合成 8–10 万候选。**

工具栈：`requests` / `arxiv` / `datasets`（拉取）、`sentence-transformers`+`faiss`（检索）、`vLLM` 或 API（LLM 合成）、`datasketch`（去重）、`jsonl/parquet`（存储）。

### 7.3 原始数据 → 训练文件：确定性模板转换

原始数据是**嵌套 JSON 标注**（管理/审计用），训练文件是**序列化文本**（建模用）。中间是模板函数：

```python
def templatize(s):
    instr = (f"【问题】{s.F}\n【行为】{s.B}\n【结构】{s.S}\n"
             f"【要求】请从一个表面无关的领域给出可迁移机制。")
    outp  = f"来源领域：{s.source_field}\n机制：{s.mechanism}\n映射：\n" + \
            "\n".join(f"- {t.p} → {t.q}：{t.rel}" for t in s.mapping)
    return {"instruction": instr, "input": "", "output": outp}
```

关键设计：instruction 只含 F/B/S，**不写入领域名**。

### 7.4 三层格式

| 层 | 格式 | 用途 |
|---|---|---|
| ① 原始标注 | 嵌套 JSON（problem/source/mapping/alignment） | 中间产物 |
| ② 训练文件 | JSONL（每行一个扁平对象）或 Parquet | 喂给框架 |
| ③ GPU 张量 | token id 序列 | 模型算 |

### 7.5 最佳数据量

| 数据 | 最佳量 | 区间 |
|---|---|---|
| A 档 SFT 主训练 | **4 万** | 3–5 万 |
| B 档 DPO 偏好对 | **1 万** | 5千–1万 |
| C 档验收集 | **500–800** | 500–1千（不参与训练） |

- 地板：1.8 万（12–15 领域 × 每域 ≥1500）。
- 上限：行为引导类任务几万条到甜点，加多边际递减。**质量 > 数量**。
- C 档必须隔离，否则「意想不到」假性通过（测的是记忆不是能力）。

### 7.6 alignment 计算

```
alignment = 0.6 × (一致关系三元组数 / 源域关系总数)      ← 结构一致性
          + 0.4 × (最长一致关系链长度 / 源域最长链长度)   ← 系统性
```

第二项逼模型对上**成体系**的关系链，防止刷「堆零散三元组」的分。

---

## 8. 核心组件：库 vs 手写的工程边界

**SFT、DPO 是现成 Trainer（写配置+数据即可）；CFG、条件 dropout、熵正则 无现成库，各是一个独立模块文件。**

| 组件 | 实现 | 载体 | 依赖库 | 手写量 |
|---|---|---|---|---|
| LoRA/QLoRA | 库 | `LoraConfig` 参数 | `peft` + `bitsandbytes` | 0 |
| SFT | 库 | 数据集 + TrainingArguments | `trl.SFTTrainer` | 配置 |
| DPO | 库 | 偏好对数据集 | `trl.DPOTrainer` | 配置 |
| 条件 dropout | 自定义 | `cond_dropout_collator.py`，继承 `DataCollatorForSeq2Seq` | `transformers` | ~40 行 |
| 熵正则 | 自定义 | `entropy_trainer.py`，继承 `Trainer` 覆写 `compute_loss` | `transformers` | ~30 行 |
| CFG 推理 | 自定义 | `cfg.py`，继承 `LogitsProcessor` | `transformers` | ~50 行 |

依赖栈：`torch`、`transformers`、`peft`、`trl`、`datasets`、`bitsandbytes`、`accelerate`、`deepspeed`。

**「几行代码还是写文件」的答案**：训练配置是 YAML 参数不是代码；数据是数据文件；三个自定义逻辑是**三个独立 .py 模块，每个一个类**（要继承基类、覆写钩子）。

CFG 工程分支：严格实现是**逐 token 双前向**（`LogitsProcessor` 内同时算有/无条件 logits 再插值），计算量翻倍；近似实现是**先采样 20 候选再 log-prob 重打分**，w 对生成过程无干预。验收门 3 测的是前者。

---

## 9. 八步训练流程

| 步骤 | 做什么 |
|---|---|
| **0 拉基座** | LLaMA-Factory 拉 R1-Distill-32B，模板 `deepseekr1` |
| **1 定格式** | 输入只写 F/B/S，输出给领域+机制+映射 |
| **2 造数据** | 拉原料 → LLM 合成 → 四道过滤，C 档隔离 |
| **3 阶段1 SFT** | LoRA rank32、lr 1e-4、3 epoch |
| **4 条件 dropout** | collator 以 p=0.15 替换条件段 |
| **5 熵正则** | `loss = CE − λ_H·H` |
| **6 阶段3 DPO** | chosen/rejected-A/rejected-B 三分类 1:1:1，dropout 继续开 |
| **7 CFG 推理** | `logP_w = logP(∅) + w·[logP(c)−logP(∅)]`，末端只硬约束不排序 |
| **8 监控验收** | 固定 prompt 集测三指标 + 四道门 |

### 9.1 关键代码

条件 dropout（collator）：

```python
NULL_TEXT = "【问题】未提供\n【行为】未提供\n【结构】未提供\n【要求】请给出一个可迁移的机制。"

class CondDropoutCollator(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, p=0.15, **kw):
        super().__init__(tokenizer=tokenizer, **kw)
        self.p = p
        self.null_ids = tokenizer(NULL_TEXT, add_special_tokens=False)["input_ids"]

    def __call__(self, features):
        for f in features:
            if random.random() >= self.p:
                continue
            labels = f["labels"]
            n_prompt = 0
            while n_prompt < len(labels) and labels[n_prompt] == -100:
                n_prompt += 1
            f["input_ids"] = self.null_ids + f["input_ids"][n_prompt:]
            f["labels"] = [-100] * len(self.null_ids) + labels[n_prompt:]
        return super().__call__(features)
```

熵正则（trainer）：

```python
class EntropyRegTrainer(Trainer):
    def __init__(self, *a, lambda_h=0.005, **kw):
        super().__init__(*a, **kw)
        self.lambda_h = lambda_h

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = model(**inputs)
        logits = out.logits[:, :-1, :].float()
        p = torch.softmax(logits, dim=-1)
        H = -(p * torch.log(p + 1e-9)).sum(-1)
        mask = (inputs["labels"][:, 1:] != -100).float()
        H = (H * mask).sum() / mask.sum().clamp(min=1)
        loss = out.loss - self.lambda_h * H          # 减号 = 最大化熵
        return (loss, out) if return_outputs else loss
```

CFG 推理（旋钮）：

```python
def cfg_logprob(model, tok, prompt, response, w=0.45):
    lc = logprob(model, tok, prompt, response)         # 有条件
    lu = logprob(model, tok, NULL_PROMPT, response)    # 无条件
    return lu + w * (lc - lu)                          # 线性插值
```

### 9.2 CFG 标量 w 档位（方向别记反）

| w | 行为 | 用途 |
|---|---|---|
| 0 | 纯无条件 | 最发散，完全不看问题 |
| **0.3–0.6** | 弱条件 | **推荐发散档（0.45）** |
| 1 | 纯条件 | vanilla 基准点 |
| 1.2–1.5 | 强化条件 | 收敛档（上限 1.5） |
| > 2 | 过度引导 | 退化：重复、不自然 |

**CFG 本质是降熵**：发散靠调小 w，收敛靠调大 w。w 是「条件强度旋钮」，不是「发散旋钮」。

### 9.3 三个工程细节（踩了白训）

1. null condition 必须用统一占位串（`【问题】未提供`），不能用空串。
2. dropout 在**所有后续阶段保持开启**（阶段3也要 p=0.15），否则成果被洗掉。
3. 阶段1 结束选「峰值多样性的 checkpoint」，不是 loss 最低的那个。

---

## 10. 目录结构与运行

```
train/
├── models/R1-Distill-Qwen-32B/        # 基座权重
├── data/                              # dataset_info.json + wis_sft.jsonl + wis_dpo.jsonl
├── custom/                            # ★ 手写三处：collator / trainer / cfg
│   ├── cond_dropout_collator.py
│   ├── entropy_trainer.py
│   └── cfg_inference.py
├── configs/                           # stage1_sft / stage2_cond_dropout / stage3_dpo
├── saves/{stage1,stage2,stage3}/      # 训练产物
└── eval/four_gates.py                 # 验收
```

```powershell
llamafactory-cli train configs/stage1_sft.yaml
llamafactory-cli train configs/stage2_cond_dropout.yaml
llamafactory-cli train configs/stage3_dpo.yaml
python eval/four_gates.py --checkpoint saves/stage3
```

**语言**：全程 Python（PyTorch + Transformers），大部分是 YAML 配置，只有 `custom/` 三处手写。

---

## 11. 验收：四道门（基准集用 C 档，绝不参与训练）

| 门 | 测什么 | 阈值 |
|---|---|---|
| **1 重定向成立** | 表面距离 + 结构对齐 | 距离 ≥ 0.6 且 对齐 ≥ 0.7 |
| **2 意想不到** | 盲猜 + 可预测性 + 成立性 | 前两项重合率 < 30%，成立 ≥ 60% |
| **3 旋钮有效** | 领域覆盖熵随 w 单调 | w↑ → 熵↓ |
| **4 不崩** | 人工评分 + 一致性 | 人工 ≥ 3.5/5，一致性 ≥ 0.6 |

**门 2 是真人实验**：盲猜（专家只看问题）、可预测性（现成 LLM 只看问题答 top-1）、成立性（专家看映射后判断）。三条合起来「意想不到」= **猜不到 + 站得住**。

验收 = 门 1、门 3 脚本算指标，门 2、门 4 真人评测。

---

## 12. 失败症状速查

| 症状 | 处方 |
|---|---|
| w=0 输出断裂 | p 上调至 0.2；确认 null 用占位串非空串 |
| w=1 对齐变差 | p 下调至 0.1 |
| 调 w 没效果 | 查 collator 是否真替换；查阶段3 是否漏 keep dropout |
| 后期领域熵下降 | 提高 λ_H；加大 rejected-A 权重；回调峰值 checkpoint |
| 来源领域扎堆 | 语料领域不均衡，回阶段0 做均衡 |
| 全是保守答案 | rejected-A 权重不够，偏好比 1:1:1 |
| 输出高熵无意义 | λ_H 降到 0.003 以下 |
| 输出重复不自然 | w 过大，压到 1.5 以内 |
| 阶段3 后发散下降 | DPO 熵压缩，确认 `−λ_H·H` 在起作用 |

---

## 13. 一句话总结

**WIS = 一份权重（DeepSeek + LoRA）+ 条件 dropout 融出两个分布 + 一个条件强度旋钮 w + 三项训练期监控 + 四道验收门。**

- **理论落点**：FBS 定输入编码、SMT 定对齐度量、C-K 定发散收敛动力学。
- **三个要害**：条件 dropout 是融合载体（没它 w 无从切换）、熵正则是全程保险（每阶段都加）、阶段3 必须继续 dropout（漏了前功尽弃）。
- **成立判据**：来源领域猜不到、映射关系站得住、旋钮调得动、人工评得高分。

---

## 14. 本仓库实现与运行（`train/wis/`）

第 10 节给的是 LLaMA-Factory 形态的通用目录；本仓库把同一套方法落成自包含的 `train/wis/` 包，不依赖 LLaMA-Factory。

### 14.1 目录

```
train/wis/
├── config.py        # ROOT 定位 + load_config 校验
├── data.py          # jsonl 读写、response/rows 校验、train/eval 隔离断言
├── collator.py      # render_prompt/render_completion/encode_sample、CondDropoutCollator、PreferenceDropoutCollator
├── entropy.py       # masked_entropy、EntropySFTTrainer、EntropyDPOTrainer（−λ_H·H）
├── cfg.py           # cfg_scores、sample_token、generate_ids（逐 token 双前向 CFG）
├── prepare.py       # grouped_split（按 problem_group 确定性切分）
├── runtime.py       # preflight（配置+数据体检）
├── evaluate.py      # blind_packet / summarize（盲评汇总）
├── infer.py         # parse_prediction（模型输出→JSON）
├── train.py         # ★ 训练入口（SFT/DPO，LoRA/QLoRA）
├── smoke.py         # ★ 本地冒烟（随机极小模型，零下载跑通训练循环）
├── configs/         # 顶层 4 个参考配置（被测试契约校验）
│   ├── local/       # 8GB 4060 调参配置（1.5B 4bit）
│   └── cloud/       # 2×5090 配置（32B：QLoRA 首选 / bf16+ZeRO-3 备选）
└── pipeline/        # ★ 原料 → 训练样本 流水线
    ├── llm.py       # OpenAI 兼容客户端（环境变量配置）
    ├── io.py        # 领域归一、去重、稳定 id
    ├── extract.py   # 原料 → 机制清单
    ├── problems.py  # 问题库 + F/B/S 规范化
    ├── pair.py      # 远域配对（字符 n-gram TF-IDF 距离）
    ├── synthesize.py# 生成映射 + alignment 打分
    ├── filter.py    # 四道门 + 切分 + 落盘
    ├── templatize.py# 确定性模板转换
    └── run.py       # 编排 CLI
```

### 14.2 环境

用 conda 的 `sam3` 环境（torch 2.10.0+cu128，已确认 RTX 4060 Laptop 8GB 可用）。训练栈装在该解释器的用户目录：

```
transformers==4.46.3  datasets==2.21.0  peft==0.13.2  trl==0.12.2  accelerate==1.1.1  pyarrow==17.0.0
```

**为什么锁这些版本**：`sam3` 包自身 pin 了 `numpy<2`，而 `datasets>=3 / pyarrow>=24 / transformers>=5` 都要求 `numpy>=2`，会与 `numpy 1.26` 冲突。上面这组是同时兼容 `numpy 1.26` 的最后一版组合，SFT 与 DPO 都能跑。

**两个已知坑**：
1. `pyarrow` 目录里带 `msvcp140.dll`，若它在 `torch` 之前被加载，会让 `c10.dll` 初始化失败（`WinError 1114`）。因此 `train.py` 里 `import torch` 必须先于 `transformers`。
2. `env` 里 `pyarrow 26` 与 `numpy 1.26` 本就不自洽（前者要 numpy≥2），用 `--user` 装的 `pyarrow 17` 覆盖之；不动 env 文件，可 `pip uninstall pyarrow` 回退。

### 14.3 流水线：原料 → 训练样本

```powershell
# 离线冒烟（不调 LLM，全链路跑通，产物结构合法）
python -m train.wis.pipeline.run --offline --limit 400 --out-dir train/wis/data/pipeline_smoke

# 真实模式（需环境变量：OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL）
python -m train.wis.pipeline.run --limit 20000 --out-dir train/wis/data `
    --seed-problems train/wis/data/examples/eval_prompts.jsonl

# 复用已有机制清单，只重跑后半程
python -m train.wis.pipeline.run --inventory train/wis/data/private/mechanisms_inventory.jsonl --limit 20000
```

阶段与中间产物：`mechanisms_inventory.jsonl → problems.jsonl → pairs.jsonl → candidates.jsonl → wis_sft.jsonl / wis_sft_eval.jsonl / wis_dpo.jsonl / wis_dpo_eval.jsonl`。

四道门在 `filter.py`：`alignment ≥ 0.7`、`表面距离 ≥ 0.6`、去重、训练/验证按 `problem_group` + `mechanism_group` 双隔离（隔离后仍调用 `assert_disjoint` 硬校验）。

### 14.4 训练

```powershell
# 本地冒烟：零下载，验证训练循环（条件 dropout / 熵正则 / LoRA / 保存）
python -m train.wis.smoke --stage both

# 本地 4060 调参（1.5B 4bit；示例数据需 --allow-demo）
python -m train.wis.train --config train/wis/configs/local/sft_4060.json --allow-demo
python -m train.wis.train --config train/wis/configs/local/dpo_4060.json --allow-demo

# 真实数据 + 换更大基座（8GB 上限约 7B-4bit）
python -m train.wis.train --config train/wis/configs/local/sft_4060.json --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# 云端 2×5090 / 32B（数据来自流水线，先跑 14.3 生成）
accelerate launch --num_processes 2 -m train.wis.train --config train/wis/configs/cloud/sft_32b.json
accelerate launch --num_processes 2 -m train.wis.train --config train/wis/configs/cloud/dpo_32b.json
```

`--stage` 默认按配置文件名推断（含 `dpo` → DPO，否则 SFT）；`--max-steps` 便于快速试跑。

### 14.5 本地已验证的结论

- 22 项离线测试全通过（`python -m pytest tests/test_offline.py`）。
- 流水线离线全链路：400 篇原料 → 2400 候选 → 1920 训练 / 15 验证（SFT 与 DPO 均过校验）。
- SFT 与 DPO 训练循环在 4060 上真实跑通并落盘 adapter（`python -m train.wis.smoke --stage both`）。
- **真实数据已生成**（`train/wis/data/real_v1/`）：3172 候选 → 3059 条过四道门 → SFT/DPO 训练集各 2760、验证集各 299；9 大领域（材料/生物/航空航天/机械/电子/计算机/能源/环境/医学）覆盖均衡，`null_preference_validated` 全为 True。
- **本地真实数据 SFT 已在 4060 上启动**（1.5B 4bit，2760 训练 / 299 验证，`sft_real.json`），loss 从 2.688 稳步下降，收敛正常；完成后自动接续 `dpo_real.json`（cond_dropout=0.15，偏好 dropout collator）。
- 结论：**代码链路 + 真实数据链路已打通**；本地调参验证后按第 15 节迁移云端。

---

## 15. 云端 2×5090 迁移运行说明

本地 4060（8GB）只能验证链路与调参，正式训练在 2×5090（32GB×2）上跑 32B 版本。以下是把本仓库搬到云端 Linux 的完整步骤。

### 15.1 硬件与前置

| 项 | 要求 |
|---|---|
| GPU | 2× RTX 5090（32GB×2 = 64GB，Blackwell **sm_120**，**无 NVLink**，多卡通信走 PCIe） |
| CPU RAM | **≥ 96GB**（仅 bf16 + ZeRO-3 CPU offload **备选**路线需要，把 32B bf16 权重 ≈ 64GB 落到内存）；**QLoRA 首选路线用不到这么多** |
| 磁盘 | ≥ 1TB NVMe（基座权重 + 数据 + checkpoint） |
| 系统 | Linux（DeepSpeed 对原生 Windows 支持差；本地 Windows 只做调参） |

### 15.2 环境

```bash
conda create -n wis python=3.10 -y && conda activate wis
# 5090 是 Blackwell（sm_120）。cu124 的预编译 kernel 不含 sm_120，运行时会直接报
# "no kernel image is available for execution on the device"——这是必踩的坑，故用 cu128。
pip install --upgrade "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128
pip install transformers==4.46.3 datasets==2.21.0 peft==0.13.2 trl \
            accelerate==1.1.1 pyarrow==17.0.0 deepspeed bitsandbytes
```

装完先确认 torch 与算力匹配（应打印 `2.x`、`12.8`、`(12, 0)`）：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_capability())"
```

云端不必锁 `numpy<2`（那是本地 `sam3` 环境的约束），但沿用同版本组可保证与本地行为一致。

⚠️ **trl 版本必须与本地一致**：本地 trl 的 `dist-info` 标 `0.12.2`，但实际代码是**新版 API**（`DPOTrainer` 先 `map(tokenize_row)` 再 collate，默认 collator 为 `trl.trainer.dpo_trainer.PreferenceCollator`，会生成 attention mask）。`train.py` 的 DPO 路径按这套 API 写死。装完依赖后**必须先跑冒烟验证**：

```bash
python -m train.wis.smoke --stage both   # 应输出 [sft]/[dpo] adapter saved to ...
```

若 `from trl.trainer.dpo_trainer import PreferenceCollator` 报 ImportError，说明装成了老 API 的 trl，需换回与本地一致的版本，否则 DPO 的 collate 与 `concatenated_forward` 不兼容（缺 `*_attention_mask` 会直接 KeyError）。

### 15.3 拉基座权重

```bash
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --local-dir train/wis/models/DeepSeek-R1-Distill-Qwen-32B
```

`configs/cloud/*.json` 里的 `model_name_or_path` 默认写的是 HF 仓库名，若已下到本地目录，改成 `train/wis/models/DeepSeek-R1-Distill-Qwen-32B` 即可（离线加载）。

### 15.4 数据

**目录约定（两边不要混）**：

- `train/wis/data/real_v2/` = **云端全量**数据目录，两个 32B 配置已指向这里。
- `train/wis/data/real_v2_smoke/` = 本地冒烟小样本目录，本地 `configs/local/*.json` 已指向它。
- 数据 schema 已更换（多机制 + solution + prerequisites），故不再用旧的 `real_v1`。实测总长 p50=1512 / p90=1716 token，**`max_length` 必须 ≥2048**（配置已设 2048）。

数据生成有两个选项，用户嫌本地直连 API 太慢，推荐选项 B。

**选项 A（保守，本地/任意机器 + DeepSeek API）**：

```bash
python -m train.wis.pipeline.run --provider openai --model <DeepSeek 模型名> \
    --limit 6000 --pairs-per-problem 6 --workers 4 \
    --out-dir train/wis/data/real_v2
```

实测生成一次约 11.4s，`--workers 4` 并发后 ≈ 3.6 万条样本 / **约 38 小时**。花 API 费，但生成质量与线上 flash 一致。

**选项 B（推荐给云端：2×5090 上用 vLLM 自起 32B）**：

```bash
# 先在云端起 OpenAI 兼容端点（TP=2 把 32B 摊到两张卡）
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --tensor-parallel-size 2 --max-model-len 4096 --port 8000

# 另开终端，把流水线指向本机端点，并发拉到 32
python -m train.wis.pipeline.run --provider openai \
    --host http://127.0.0.1:8000/v1 --model <vLLM 上报的模型名> \
    --limit 6000 --pairs-per-problem 6 --workers 32 \
    --out-dir train/wis/data/real_v2
```

`--model` 填 vLLM `/v1/models` 上报的服务名，`--host` 指向本机端点（默认 Ollama 的 11434 端口不适用）。并行 32 下预估 **约 1–2 小时量级**，需实测校准。

**选项取舍**：B 快且不花 API 费，但要额外装 vLLM，且 32B 本地生成的样本质量与线上 flash 不同，**必须抽检**（对比选项 A 的少量样本）再决定是否全量采用。

生成/拷入后必备文件：`wis_sft.jsonl`、`wis_sft_eval.jsonl`、`wis_dpo.jsonl`、`wis_dpo_eval.jsonl`（另带 `domain_distance.json`、`mechanisms_inventory.jsonl` 供追溯）。若数据在别的机器生成，整目录拷过去即可，**无需在训练机上重跑流水线**：

```bash
rsync -av train/wis/data/real_v2/ user@cloud:/path/Innomap-master/train/wis/data/real_v2/
```

### 15.5 训练

```bash
# 阶段1：SFT（32B nf4 QLoRA + LoRA r64，免 offload）
accelerate launch --num_processes 2 -m train.wis.train \
    --config train/wis/configs/cloud/sft_32b.json --stage sft

# 阶段3：DPO（init_adapter 自动从 cloud-sft-32b/adapter 续接）
accelerate launch --num_processes 2 -m train.wis.train \
    --config train/wis/configs/cloud/dpo_32b.json --stage dpo
```

`--stage` 可省略（按配置文件名推断）；快速试跑加 `--max-steps 20` 即可。两个阶段均设 `num_train_epochs: 1`——数据扩量后 1 epoch 已足够，避免过拟合。

**默认走 QLoRA（首选）**：两个配置都是 `load_in_4bit: true`——权重按 nf4 + double quant 加载、compute 用 bf16（沿用 `train.py::_load_model` 里已有的 `BitsAndBytesConfig` 写法），且**已移除 `deepspeed` 键**，直接靠 `accelerate` 多卡 DDP 即可。这样可完全免掉 offload，规避 5090 **无 NVLink** 时 ZeRO-3 all-gather 走 PCIe 成为主瓶颈的问题（DPO 因熵正则每步还要多跑一次全量 forward，更雪上加霜）。

**备选：bf16 全精度 + ZeRO-3 CPU offload**。如需该路线，把两个配置的 `load_in_4bit` 改回 `false`，并加回 `"deepspeed": "train/wis/configs/cloud/ds_zero3.json"`。该 DeepSpeed 文件本身有效（stage 3、`offload_optimizer`/`offload_param` 为 cpu、`pin_memory: true`、`overlap_comm: true`、`bf16` enabled；`train_batch_size`/micro/accum 均为 `auto`，由 TrainingArguments 填充），无需再写 `accelerate config` 的 deepspeed 段。此路线要求宿主 RAM ≥96GB。

### 15.6 显存预算与 OOM 处置

**QLoRA 首选路线（推荐，免 offload）**：

- 32B nf4 权重 ≈ 19GB；2×5090 合计 64GB，权重 + LoRA + 激活仍远低于总显存 → **完全不需要 offload**。
- LoRA 只训约 160M 参数（rank 64，≈0.5%），`paged_adamw_8bit` 优化器状态极小；显存瓶颈在激活而非优化器。
- 开 `gradient_checkpointing: true` 后，`max_length 2048`、SFT micro-batch 2 / DPO micro-batch 1 均可稳住。
- **降级顺序**：① `max_length` 2048 → 1536；② micro-batch 降到 1 并翻倍 `gradient_accumulation_steps`（保持等效 batch）；③ 换 14B/7B。

**bf16 全精度 + ZeRO-3 备选路线**：

- 32B bf16 权重 = 64GB，恰好等于 2×5090 总显存 → **必须开 ZeRO-3 offload**（param + optimizer 落 CPU，宿主 RAM ≥96GB）。
- **降级顺序**：① 确认 `offload_param` / `offload_optimizer` 已开；② `max_length` 2048 → 1024；③ CPU RAM 不足 96GB 时改 ZeRO-2（只 offload optimizer，权重留 GPU）；④ 仍不够则退 14B/7B 或用 4 卡。
- 5090 支持 bf16，两条路线 `bf16: true` 均保持开启。

### 15.7 产出与验收

- 产物：`train/wis/outputs/cloud-sft-32b/adapter`、`train/wis/outputs/cloud-dpo-32b/adapter`（LoRA adapter，非独立大模型）。
- 拷回本地或直接在云端用 `cfg.py` 做 CFG 推理（w=0.45 发散档），跑第 11 节四道门验收。
- 三阶段产出的 adapter 必须成对保存；DPO 若换机器续训，务必带上 `cloud-sft-32b/adapter` 作为 `init_adapter`，否则会从零重训。