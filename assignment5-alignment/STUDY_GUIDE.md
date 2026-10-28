# Assignment 5: Alignment & Reasoning RL — 详细通关实战指南

> **本目录目标**：掌握现代大模型后训练（Post-Training）与强化学习推理对齐全流程：
> 1. **主线任务（Reasoning RL / DeepSeek-R1 风格）**：基于可验证数学奖励（Verifiable Rewards）实现 **SFT + GRPO / Dr.GRPO / MaxRL / GSPO** 强化学习训练流水线；
> 2. **补充任务（Safety & RLHF）**：实现指令微调数据集打包（Packed SFT Dataset）、MMLU/GSM8K 评测解析器与 **DPO（Direct Preference Optimization）** 直接偏好优化。

---

## 📁 推荐你在 `cs336_alignment/` 下建立的工程结构

```text
assignment5-alignment/
├── cs336_alignment/
│   ├── __init__.py
│   ├── drgrpo_grader.py        # [自带] R1-Zero 格式与数学答案校验器 (r1_zero_reward_fn)
│   ├── vllm_utils.py           # [自带] vLLM 高速采样辅助工具
│   ├── sft_utils.py            # [新建-模块 1] tokenize_prompt_and_output, get_response_log_probs, PackedSFTDataset
│   ├── grpo.py                 # [新建-模块 2] compute_rollout_rewards, compute_group_normalized_rewards,
│   │                           #               compute_policy_gradient_loss, aggregate_loss_across_microbatch, grpo_train_step
│   ├── metrics.py              # [新建-模块 3] parse_mmlu_response, parse_gsm8k_response
│   └── dpo.py                  # [新建-模块 4] compute_per_instance_dpo_loss
└── tests/
    └── adapters.py             # 连接上述模块与 pytest 单元测试
```

---

## 前置准备：看课与精读 PDF

- [ ] **看课**：
  - 观看 **Lecture 15 (Alignment I: SFT & RLHF)**，阅读课件 [../lectures/lecture_15.pdf](../lectures/lecture_15.pdf)。
  - 观看 **Lecture 16 (Alignment II: DPO & GRPO)**，阅读课件 [../lectures/lecture_16.pdf](../lectures/lecture_16.pdf)。
- [ ] **读 PDF**：
  - 主文档：[cs336_spring2026_assignment5_alignment.pdf](./cs336_spring2026_assignment5_alignment.pdf)
  - 补充文档：[cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)

---

## 模块 1：Prompt-Response 分词掩码与对数概率计算 (`cs336_alignment/sft_utils.py`)

### 1. 实现 `run_tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer)`
- **读主 PDF**：Section 3 (SFT / Tokenization) 及 [tests/adapters.py](./tests/adapters.py#L13-L49) 的 docstring。
- **核心逻辑**：
  1. 分别对每个 `prompt` 和 `output` 独立编码（不加额外 special tokens，或按 `tokenizer.encode(s, add_special_tokens=False)`），设长度分别为 $L_p$ 和 $L_o$，拼接后完整序列长度为 $L = L_p + L_o$。
  2. 找出 batch 内最大总长度 $L_{\max} = \max(L)$，用 `tokenizer.pad_token_id` 在右侧填充到 $L_{\max}$。
  3. 构造自回归错位输入输出（形状均为 `(batch_size, L_max - 1)`）：
     - `input_ids = full_ids[:, :-1]`
     - `labels = full_ids[:, 1:]`
  4. 构造与 `labels` 对齐的 `response_mask`（形状 `(batch_size, L_max - 1)`）：
     - 在 `labels` 中，下标范围 `[L_p - 1, L_p + L_o - 1)` 对应的是 `output` 的各个 token，将这些位置设为 `1`（或 `True`），其余位置（Prompt 区域和 Padding 区域）设为 `0`！
- [ ] **单测验收**：
  ```bash
  uv run pytest tests/test_grpo.py -k "test_tokenize_prompt_and_output"
  ```

### 2. 实现 `run_get_response_log_probs(model, input_ids, labels, return_token_entropy)`
- **核心逻辑**：
  1. 前向计算 `logits = model(input_ids).logits`（形状 `(B, T, V)`）。
  2. 计算全词表对数概率：`log_probs_all = torch.log_softmax(logits, dim=-1)`。
  3. 用 `torch.gather(log_probs_all, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)` 取出每个位置真实 `labels` 对应的条件对数概率 `log_probs`（形状 `(B, T)`）。
  4. 若 `return_token_entropy=True`：
     计算每个位置的香农熵 $H(p) = -\sum_{v=1}^V p_v \log p_v = -\sum_{v=1}^V \exp(\text{log\_probs\_all}) \cdot \text{log\_probs\_all}$，沿 `dim=-1` 求和得到 `token_entropy`（形状 `(B, T)`）。
- [ ] **单测验收**：
  ```bash
  uv run pytest tests/test_grpo.py -k "test_get_response_log_probs"
  ```

---

## 模块 2：GRPO 及其前沿变体 (Dr.GRPO / MaxRL / GSPO) (`cs336_alignment/grpo.py`)

### 1. 实现 `run_compute_rollout_rewards(reward_fn, rollout_responses, repeated_ground_truths)`
- 遍历 `zip(rollout_responses, repeated_ground_truths)`，调用 `reward_fn(resp, gt)` 得到包含 `"reward"`, `"format_reward"`, `"answer_reward"` 的字典。
- 将所有 `"reward"` 组装为 `torch.float32` 的一维张量 `raw_rewards`（形状 `(rollout_batch_size,)`），并在 `metadata` 中返回平均总奖励和平均格式奖励。
- [ ] **单测验收**：`uv run pytest tests/test_grpo.py -k "test_compute_rollout_rewards"`

### 2. 实现组内归一化优势 `run_compute_group_normalized_rewards`
- 将 `raw_rewards` 重塑为 `(n_prompts, group_size)`：
  1. **计算基线 (`baseline`)**：
     - `"mean"`：减去组内均值 `r - r.mean(dim=-1, keepdim=True)`；
     - `"none"`：不减均值。
  2. **计算归一化分母 (`advantage_normalizer`)**：
     - `"std"`（标准 GRPO）：除以 `r.std(dim=-1, keepdim=True) + advantage_eps`；
     - `"none"`（Dr.GRPO，消除长度/方差偏差）：不除；
     - `"mean"`（MaxRL）：除以 `r.mean(dim=-1, keepdim=True) + advantage_eps`。
  3. 展平回 `(rollout_batch_size,)` 返回 `(advantages, metadata)`。
- [ ] **单测验收**：`uv run pytest tests/test_grpo.py -k "test_compute_group_normalized_rewards"`

### 3. 实现策略梯度损失 `run_compute_policy_gradient_loss`
- 统一将 `A = raw_rewards_or_advantages` 整理为 `(batch_size, 1)` 以便在序列维度广播。
- 支持 4 种 `importance_reweighting_method`：
  1. **`"none"`（On-Policy REINFORCE）**：
     $$\ell_{b, t} = -A_b \cdot \log \pi_\theta(y_{b,t} \mid x_b, y_{b,<t})$$
  2. **`"noclip"`（Off-Policy 无裁剪重要性采样）**：
     设比率 $\rho_{b,t} = \exp(\text{policy\_log\_probs}_{b,t} - \text{old\_log\_probs}_{b,t})$，则 $\ell_{b,t} = -\rho_{b,t} A_b$。
  3. **`"grpo"`（Token 级 PPO/GRPO 裁剪）**：
     $$\ell_{b,t} = -\min\Big(\rho_{b,t} A_b,\; \text{clip}(\rho_{b,t}, 1 - \epsilon, 1 + \epsilon) A_b\Big)$$
  4. **`"gspo"`（GSPO 序列级几何平均重要性采样裁剪）**：
     先在每个序列的有效 `response_mask` 范围内计算**平均对数概率比**：
     $$\bar{\delta}_b = \frac{\sum_t \text{mask}_{b,t} (\log \pi_{\theta, b, t} - \log \pi_{\text{old}, b, t})}{\sum_t \text{mask}_{b,t}}$$
     序列级比率 $\rho_b^{\text{seq}} = \exp(\bar{\delta}_b)$，再用 $\rho_b^{\text{seq}}$ 与 $\text{clip}(\rho_b^{\text{seq}}, 1-\epsilon, 1+\epsilon)$ 计算损失！
- [ ] **单测验收**：`uv run pytest tests/test_grpo.py -k "test_compute_policy_gradient_loss"`

### 4. 实现损失聚合 `run_aggregate_loss_across_microbatch` 与训练步 `run_grpo_train_step`
- **`loss_normalization == "sequence"`**：先对每个样本在 `mask == 1` 的 token 上求平均 $\frac{\sum_t \text{loss}_{b,t} \cdot \text{mask}_{b,t}}{\sum_t \text{mask}_{b,t}}$，再对 batch 维求 `.mean()`。
- **`loss_normalization == "constant"`**：直接将整个 microbatch 内所有有效位置损失之和除以固定常数 `normalization_constant`（再除以 batch_size，请对照快照测试验证具体系数）。
- **`run_grpo_train_step`**：串联打分 $\to$ 组归一化 $\to$ 分词 $\to$ 按 `gradient_accumulation_steps` 切分 microbatch $\to$ 每个 microbatch 前向算 `policy_log_probs`、算 Loss、除以 `gradient_accumulation_steps` 后 `.backward()` $\to$ 可选 `clip_grad_norm_` $\to$ `optimizer.step()`。
- [ ] **单测验收**：
  ```bash
  uv run pytest tests/test_grpo.py
  ```

---

## 模块 3：SFT 打包数据集与评测解析器 (`sft_utils.py` & `metrics.py`)

### 1. 实现 `get_packed_sft_dataset` 与 `run_iterate_batches`
- 读取 `dataset_path`（JSONL 格式，含 `prompt` 和 `response`），使用 Alpaca 模板格式化每条文档（若 `shuffle=True` 则先打乱文档顺序），在文档间拼接 `bos_token_id` / `eos_token_id` 形成一条长一维 Token 流。
- 按固定步长 `seq_length` 切块，构造 `"input_ids"` 和右移一位的 `"labels"`（对照 [tests/test_data.py](./tests/test_data.py) 中的 `tokenized_sft_sample.json` 确认 `bos`/`eos` 拼接细节）。
- [ ] **单测验收**：`uv run pytest tests/test_data.py`

### 2. 实现 `run_parse_mmlu_response` 与 `run_parse_gsm8k_response`
- **MMLU**：从模型回答中用正则提取选项字母 `"A"`, `"B"`, `"C"`, `"D"`，无法解析返回 `None`。
- **GSM8K**：用正则 `r"[-+]?\d[\d,]*(?:\.\d+)?"` 提取文本中出现的**最后一个数字**（去掉千位分隔符逗号 `,`），若无数字则返回 `None`。
- [ ] **单测验收**：`uv run pytest tests/test_metrics.py`

---

## 模块 4：DPO 直接偏好优化 (`cs336_alignment/dpo.py`)

- [ ] **读补充 PDF**：DPO 损失推导章节，并查看 [tests/test_dpo.py](./tests/test_dpo.py)。
- [ ] **实现 `run_compute_per_instance_dpo_loss`**：
  1. 用 Alpaca 模板将 `(prompt, response_chosen)` 与 `(prompt, response_rejected)` 格式化并追加 `eos_token`。
  2. 分别计算策略模型 `lm` 和参考模型 `lm_ref` 在完整序列（或 response 序列）上的对数概率之和 $\log \pi_\theta(y_w \mid x)$、$\log \pi_\theta(y_l \mid x)$、$\log \pi_{\text{ref}}(y_w \mid x)$、$\log \pi_{\text{ref}}(y_l \mid x)$。
  3. 代入 DPO 单样本公式：
     $$\mathcal{L}_{\text{DPO}} = -\text{F.logsigmoid}\Big(\beta \big[\log \pi_\theta(y_w|x) - \log \pi_{\text{ref}}(y_w|x)\big] - \beta \big[\log \pi_\theta(y_l|x) - \log \pi_{\text{ref}}(y_l|x)\big]\Big)$$
- [ ] **单测验收**：
  ```bash
  uv run pytest tests/test_dpo.py
  ```
- [ ] **Assignment 5 全量单测总验收**：
  ```bash
  uv run pytest
  ```
