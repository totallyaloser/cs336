# Assignment 1: Basics — 详细通关实战指南

> **本目录目标**：不依赖 `transformers` 或 `torch.nn` 现成的高层层模块（如 `nn.Linear`、`nn.RMSNorm`、`nn.MultiheadAttention`、`F.softmax`、`F.cross_entropy`），纯手写实现 **BPE 分词器 + LLaMA 风格现代 Transformer 语言模型 + AdamW 优化器 + 完整训练与推理闭环**。

---

## 📁 推荐你在 `cs336_basics/` 下建立的工程结构

不要把所有代码都堆在 `tests/adapters.py` 里！建议在 `cs336_basics/` 下按职责新建以下文件，然后在 `tests/adapters.py` 中仅做 `import` 和转发：

```text
assignment1-basics/
├── cs336_basics/
│   ├── __init__.py
│   ├── bpe.py              # [阶段 1] train_bpe 函数（可参考 tests/yu/bpe_optimized_ref.py）
│   ├── tokenizer.py        # [阶段 2] Tokenizer 类（encode / decode / encode_iterable / from_files）
│   ├── nn_utils.py         # [阶段 3 & 4] softmax, silu, cross_entropy, gradient_clipping
│   ├── model.py            # [阶段 3] Linear, Embedding, RMSNorm, SwiGLU, RoPE, Attention, Block, TransformerLM
│   ├── optimizer.py        # [阶段 4] AdamW 类, get_lr_cosine_schedule 函数
│   ├── data.py             # [阶段 5] get_batch, save_checkpoint, load_checkpoint
│   ├── train.py            # [阶段 6] 完整训练主循环脚本（支持 CLI 参数与日志记录）
│   └── decode.py           # [阶段 6] 文本生成脚本（Temperature + Top-p 采样）
└── tests/
    └── adapters.py         # 仅在此文件中 import 上述模块并调用
```

---

## 第一阶段：BPE 分词器训练 (`cs336_basics/bpe.py`)

### 1. 学习准备
- [ ] **看课**：Lecture 1 视频 + 阅读 [../lectures/lecture_01.py](../lectures/lecture_01.py) 中的 `tokenization()` 部分。
- [ ] **读 PDF**：[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf) **Section 2.1 ~ 2.5 (第 2 ~ 9 页)**。
- [ ] **完成书面小题**：
  - `Problem (unicode1)`：`chr(0)` 是什么字符？它的 `__repr__` 和 `print` 有何区别？
  - `Problem (unicode2)`：为什么 BPE 选 UTF-8 而不是 UTF-16/UTF-32？构造一个 UTF-8 多字节字符（如中文 `"你好"`）说明逐字节 `decode` 为什么会报错。

### 2. 代码实现与优化清单
- [x] **基础版实现**：已在 [tests/adapters.py](./tests/adapters.py) 完成并通过 3 个测试。
- [ ] **迁移并升级为多进程 + 最大堆优化版**：
  - 参考为你写好的对照文件 [tests/yu/bpe_optimized_ref.py](./tests/yu/bpe_optimized_ref.py)，将核心逻辑整理到 `cs336_basics/bpe.py`。
  - **必查 3 个易错点**：
    1. `special_tokens` 为空列表 `[]` 时，不能调 `re.split("", text)`，否则会把字符串按单字符切碎；
    2. 初始词表包含 256 个单字节 `bytes([0..255])` + `special_tokens` 编码后的 `bytes`，总合并次数为 `vocab_size - 256 - len(special_tokens)`；
    3. 平局决胜规则（Tie-breaking）：频次相同时，选元组 `(part1, part2)` **字典序较大**的合并。
- [ ] **单测验收**：
  ```bash
  uv run pytest tests/test_train_bpe.py
  ```

### 3. 数据集实战实验（Section 2.5）
- [ ] **实验 1 (`train_bpe_tinystories`)**：
  - 在 `data/TinyStoriesV2-GPT4-train.txt` (2.1GB) 上训练 `vocab_size=10000, special_tokens=["<|endoftext|>"]`。
  - 将输出的 `vocab` 和 `merges` 用 `pickle` 或 `json` 保存到磁盘（例如 `data/tinystories_bpe_10k.pkl`）。
  - 记录耗时与内存峰值，用 `max(vocab.values(), key=len)` 查看最长的 token 是什么。
  - 使用 `python -m cProfile -s cumtime` 分析哪一步最耗时。
- [ ] **实验 2 (`train_bpe_expts_owt`)**：
  - 在 `data/owt_train.txt` (12GB) 上训练 `vocab_size=32000, special_tokens=["<|endoftext|>"]` 并保存。

---

## 第二阶段：BPE 分词器编解码 (`cs336_basics/tokenizer.py`)

### 1. 学习准备
- [ ] **读 PDF**：[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf) **Section 2.6 ~ 2.7 (第 10 ~ 13 页)**。

### 2. 核心实现细节（必看避坑指南）
在 `cs336_basics/tokenizer.py` 中实现 `class Tokenizer`：

1. **`__init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None)`**：
   - 建立反向查表字典：`self.bytes_to_id: dict[bytes, int] = {v: k for k, v in vocab.items()}`。
   - 建立合并优先级字典：`self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: idx for idx, pair in enumerate(merges)}`。
   - 如果传入的 `special_tokens` 中有还不在 `vocab` 里的，将其追加进 `self.vocab` 和 `self.bytes_to_id`。
   - **关键点（重叠特殊 Token）**：构建切分正则时，必须先将 `special_tokens` **按长度从长到短排序**，并用**带捕获组的括号**包裹：
     ```python
     sorted_specials = sorted(special_tokens, key=len, reverse=True)
     self.special_rx = re.compile("(" + "|".join(re.escape(s) for s in sorted_specials) + ")")
     ```
     这样 `self.special_rx.split(text)` 切出来的列表中，特殊 token 本身也会被保留下来！

2. **`_encode_word(self, word_bytes: bytes) -> list[int]`（单个预分词块的 BPE 合并）**：
   - 将 `word_bytes` 拆成单字节列表 `parts = [bytes([b]) for b in word_bytes]`。
   - 循环查找当前 `parts` 所有相邻对 `(parts[i], parts[i+1])` 中，在 `self.merge_ranks` 里 **rank 最小（即最先被合并）** 的那个 pair：
     ```python
     best_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float("inf")))
     if best_pair not in self.merge_ranks:
         break
     ```
   - 将 `parts` 中所有 `best_pair` 合并为 `best_pair[0] + best_pair[1]`，重复直到无可合并项，最后查 `self.bytes_to_id` 转成 `list[int]`。
   - 💡 **性能优化技巧**：加一个字典缓存 `self._word_cache: dict[bytes, list[int]] = {}`！因为自然语言中大量单词重复出现，缓存每个预分词块的合并结果可以让 `encode` 速度提升 **10倍以上**！

3. **`encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]`（流式生成器）**：
   - `test_encode_iterable_memory_usage` 会严格限制内存上限！
   - **绝对不能**把 `iterable` 拼成大字符串或把结果存成大 `list` 返回，必须用 `for line in iterable:` 配合 `yield from self.encode(line)` 做惰性生成！

4. **`decode(self, ids: list[int]) -> str`**：
   - `b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")`。

### 3. 单测验收
- [ ] 在 [tests/adapters.py](./tests/adapters.py) 中对接 `get_tokenizer`，运行：
  ```bash
  uv run pytest tests/test_tokenizer.py
  ```
  *(共 24 个测试用例，包括重叠特殊 token、德语、地址、与 tiktoken 对齐、流式内存限制测试)*

---

## 第三阶段：Transformer 模型架构实现 (`cs336_basics/model.py` & `nn_utils.py`)

### 1. 学习准备
- [ ] **看课**：Lecture 2 ([../lectures/lecture_02.py](../lectures/lecture_02.py)) + Lecture 3 ([../lectures/lecture_03.pdf](../lectures/lecture_03.pdf))。
- [ ] **读 PDF**：[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf) **Section 3.1 ~ 3.7 (第 13 ~ 26 页)**。

### 2. 逐个组件实现指南与公式对照（按顺序写，写一个测一个）

#### 组件 3.1：`Linear` 层（Section 3.4.1）
- **要求**：继承 `nn.Module`，参数命名为 `self.weight`（形状 `(d_out, d_in)`，即 `nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))`），**无 bias**。
- **初始化**：$\sigma = \sqrt{\frac{2}{d_{\text{in}} + d_{\text{out}}}}$，使用 `nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)`。
- **前向**：`einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")` 或 `x @ self.weight.T`。
- [ ] 对接 `run_linear`，测试：`uv run pytest tests/test_model.py -k "test_linear"`

#### 组件 3.2：`Embedding` 层（Section 3.4.2）
- **要求**：参数 `self.weight` 形状 `(num_embeddings, embedding_dim)`，初始化 `trunc_normal_(mean=0.0, std=1.0, a=-3.0, b=3.0)`。
- **前向**：直接张量索引 `self.weight[token_ids]`。
- [ ] 对接 `run_embedding`，测试：`uv run pytest tests/test_model.py -k "test_embedding"`

#### 组件 3.3：`RMSNorm` 层（Section 3.5.1）
- **公式**：
  $$\text{RMSNorm}(x_i) = \frac{x_i}{\sqrt{\frac{1}{d_{\text{model}}} \sum_{j=1}^{d_{\text{model}}} x_j^2 + \epsilon}} \cdot g_i$$
- **避坑点**：计算前先保留 `in_dtype = x.dtype`，将 `x = x.to(torch.float32)` 算完归一化后再 `.to(in_dtype) * self.weight`。
- [ ] 对接 `run_rmsnorm`，测试：`uv run pytest tests/test_model.py -k "test_rmsnorm"`

#### 组件 3.4：`silu` 与 `SwiGLU` 层（Section 3.5.2）
- **公式**：
  $$\text{SiLU}(x) = x \cdot \text{sigmoid}(x), \quad \text{FFN}_{\text{SwiGLU}}(x) = W_2\big(\text{SiLU}(W_1 x) \odot W_3 x\big)$$
  其中 $W_1, W_3 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$，$W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$。
- [ ] 对接 `run_silu` 和 `run_swiglu`，测试：`uv run pytest tests/test_model.py -k "test_silu or test_swiglu"`

#### 组件 3.5：`RotaryPositionalEmbedding (RoPE)`（Section 3.5.3）
- **公式**：对第 $i$ 个位置、第 $k \in \{0, \dots, d_k/2 - 1\}$ 组二维分量 $(x_{2k}, x_{2k+1})$，旋转角 $\theta_{i, k} = \frac{i}{\Theta^{2k / d_k}}$：
  $$\begin{pmatrix} x'_{2k} \\ x'_{2k+1} \end{pmatrix} = \begin{pmatrix} \cos\theta_{i,k} & -\sin\theta_{i,k} \\ \sin\theta_{i,k} & \cos\theta_{i,k} \end{pmatrix} \begin{pmatrix} x_{2k} \\ x_{2k+1} \end{pmatrix}$$
- **实现技巧**：
  ```python
  freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k))
  positions = torch.arange(max_seq_len, dtype=torch.float32)
  angles = torch.outer(positions, freqs)  # (max_seq_len, d_k // 2)
  self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
  self.register_buffer("sin_cached", torch.sin(angles), persistent=False)
  ```
  前向时用 `token_positions` 索引 `cos` 和 `sin`，将 `x` 的偶数位 `x1 = x[..., 0::2]` 与奇数位 `x2 = x[..., 1::2]` 变换为 `(x1 * cos - x2 * sin, x1 * sin + x2 * cos)`，再用 `torch.stack([out1, out2], dim=-1).flatten(-2)` 还原形状！
- [ ] 对接 `run_rope`，测试：`uv run pytest tests/test_model.py -k "test_rope"`

#### 组件 3.6：`softmax` 与 `scaled_dot_product_attention`（Section 3.5.4）
- **`softmax(x, dim)`**：`x_shifted = x - x.max(dim=dim, keepdim=True).values`，再算 `exp(x_shifted) / exp(x_shifted).sum(dim=dim, keepdim=True)`。
- **`scaled_dot_product_attention(Q, K, V, mask=None)`**：
  - `scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)`
  - 注意：`mask` 为布尔张量，`True` 表示**允许关注**，`False` 表示**遮蔽**！所以用 `scores = scores.masked_fill(~mask, float("-inf"))`。
- [ ] 对接 `run_softmax` 和 `run_scaled_dot_product_attention`，测试：
  ```bash
  uv run pytest tests/test_nn_utils.py -k "test_softmax"
  uv run pytest tests/test_model.py -k "scaled_dot_product_attention"
  ```

#### 组件 3.7：`MultiHeadSelfAttention`（Section 3.5.5）
- **投影与分头**：用 `q_proj`, `k_proj`, `v_proj`, `output_proj` 四个 `Linear(d_model, d_model)`。
- 将 `Q, K, V` 从 `(... seq_len, d_model)` 重排为 `(... num_heads, seq_len, d_k)`（其中 `d_k = d_model // num_heads`）。
- **加 RoPE 的时机**：分头之后、算 Attention 之前，只对 `Q` 和 `K` 调用 `rope(Q, token_positions)` 和 `rope(K, token_positions)`（如果未传入 `token_positions`，默认用 `torch.arange(seq_len, device=x.device)`）。
- **因果掩码**：`causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))`。
- [ ] 对接 `run_multihead_self_attention` 和 `run_multihead_self_attention_with_rope`，测试：
  ```bash
  uv run pytest tests/test_model.py -k "multihead"
  ```

#### 组件 3.8：`TransformerBlock` 与 `TransformerLM`（Section 3.6 & 3.7）
- **`TransformerBlock`**：
  ```python
  x = x + self.attn(self.ln1(x), token_positions=token_positions)
  x = x + self.ffn(self.ln2(x))
  return x
  ```
- **`TransformerLM`**：`token_embeddings` $\to$ `layers` (`nn.ModuleList` 包含 `num_layers` 个 `TransformerBlock`) $\to$ `ln_final` (`RMSNorm`) $\to$ `lm_head` (`Linear(d_model, vocab_size)`，直接输出未归一化的 logits)。
- 💡 **提示**：查看 `tests/test_model.py` 中 `ts_state_dict` 的键名（如 `layers.0.attn.q_proj.weight`、`layers.0.ln1.weight`、`layers.0.ffn.w1.weight`），如果你的模块属性名与 `ts_state_dict` 一致，直接 `model.load_state_dict(weights)` 就能一行加载测试权重！
- [ ] 对接 `run_transformer_block` 和 `run_transformer_lm`，跑通全部模型测试：
  ```bash
  uv run pytest tests/test_model.py
  ```

---

## 第四阶段：损失函数、优化器与学习率调度 (`cs336_basics/optimizer.py` & `nn_utils.py`)

### 1. 学习准备
- [ ] **读 PDF**：[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf) **Section 4.1 ~ 4.5 (第 26 ~ 31 页)**。

### 2. 核心实现细节

1. **数值稳定交叉熵 `cross_entropy(inputs, targets)`（Section 4.1）**：
   - `inputs` 形状 `(..., vocab_size)`，`targets` 形状 `(...)`。
   - 先减去每行最大值：$m = \max_c z_c$，则：
     $$\ell = -\big(z_{\text{target}} - m - \log \sum_c \exp(z_c - m)\big)$$
   - 返回所有样本的平均标量 Loss：`.mean()`。
   - [ ] 对接 `run_cross_entropy`，测试：`uv run pytest tests/test_nn_utils.py -k "test_cross_entropy"`

2. **手写 `AdamW` 优化器（Section 4.3）**：
   - 继承 `torch.optim.Optimizer`。
   - 在 `step()` 中遍历 `self.param_groups` 和每个 `p`（跳过 `p.grad is None`）：
     - 从 `self.state[p]` 中读取或初始化步数 `t`（从 `1` 开始！）、一阶矩 `m = torch.zeros_like(p)`、二阶矩 `v = torch.zeros_like(p)`。
     - 更新矩：$m \leftarrow \beta_1 m + (1 - \beta_1) g$，$v \leftarrow \beta_2 v + (1 - \beta_2) g^2$。
     - 计算带偏置修正的步长：$\alpha_t = \text{lr} \cdot \frac{\sqrt{1 - \beta_2^t}}{1 - \beta_1^t}$。
     - 参数更新：$p \leftarrow p - \alpha_t \frac{m}{\sqrt{v} + \epsilon}$，随后施加解耦权重衰减：$p \leftarrow p - \text{lr} \cdot \lambda \cdot p$。
   - [ ] 对接 `get_adamw_cls`，测试：`uv run pytest tests/test_optimizer.py -k "test_adamw"`

3. **余弦退火学习率调度 `get_lr_cosine_schedule`（Section 4.4）**：
   - 若 $t < T_w$（Warmup 阶段）：$\alpha_t = \frac{t}{T_w} \alpha_{\max}$。
   - 若 $T_w \le t \le T_c$（余弦退火阶段）：
     $$\alpha_t = \alpha_{\min} + \frac{1}{2}\left(1 + \cos\left(\frac{t - T_w}{T_c - T_w}\pi\right)\right)(\alpha_{\max} - \alpha_{\min})$$
   - 若 $t > T_c$（退火结束）：$\alpha_t = \alpha_{\min}$。
   - [ ] 对接 `run_get_lr_cosine_schedule`，测试：`uv run pytest tests/test_optimizer.py -k "test_get_lr_cosine_schedule"`

4. **全局梯度裁剪 `gradient_clipping(parameters, max_l2_norm)`（Section 4.5）**：
   - 收集所有非空梯度 `[p.grad for p in parameters if p.grad is not None]`。
   - 计算全局 L2 范数：$\|g\|_2 = \sqrt{\sum_p \|g_p\|_2^2}$。
   - 若 $\|g\|_2 > M$，则原地缩放每个梯度：`g.mul_(M / (total_norm + 1e-6))`。
   - [ ] 对接 `run_gradient_clipping`，测试：`uv run pytest tests/test_nn_utils.py -k "test_gradient_clipping"`

---

## 第五阶段：数据批处理与 Checkpoint 序列化 (`cs336_basics/data.py`)

### 1. 学习准备
- [ ] **读 PDF**：[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf) **Section 5.1 ~ 5.2 (第 31 ~ 33 页)**。

### 2. 核心实现细节
1. **`get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str)`**：
   - 随机采样 `batch_size` 个起点下标 $i \in [0, \text{len}(\text{dataset}) - \text{context\_length} - 1]$（即 `torch.randint(0, len(dataset) - context_length, (batch_size,))`）。
   - 构造输入序列 `x = [dataset[i : i + context_length] ...]` 和右移一位的目标序列 `y = [dataset[i + 1 : i + context_length + 1] ...]`，转为 `torch.long` 并放到 `device` 上返回 `(x, y)`。
   - [ ] 对接 `run_get_batch`，测试：`uv run pytest tests/test_data.py`

2. **`save_checkpoint` 与 `load_checkpoint`**：
   - 保存字典：`{"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}`，调用 `torch.save(obj, out)`。
   - 加载时调用 `torch.load(src)`，恢复 `model.load_state_dict(...)` 和 `optimizer.load_state_dict(...)`，返回 `iteration`。
   - [ ] 对接 `run_save_checkpoint` 和 `run_load_checkpoint`，测试：`uv run pytest tests/test_serialization.py`

3. **全量单测总验收**：
   ```bash
   uv run pytest
   ```

---

## 第六阶段：在 RTX 5090 D 上跑完整训练与消融实验 (`train.py` & `decode.py`)

- [ ] **读 PDF**：**Section 5.3、Section 6 (Generating Text) 与 Section 7 (Experiments，第 33 ~ 40 页)**。
- [ ] **实现 `decode.py`**：支持温度缩放 $\text{softmax}(z / \tau)$、Top-p (Nucleus) 截断采样，遇到 `<|endoftext|>` 停止生成。
- [ ] **训练 TinyStories 基线模型**：
  - 参数配置：`vocab_size=10000, context_length=256, d_model=512, d_ff=1344, num_layers=4, num_heads=16, rope_theta=10000`（约 17M 参数，在你的 RTX 5090 D 上使用 BF16 仅需几分钟即可跑完 400M tokens！）。
  - 目标：验证集 Loss 降到 **$\le 1.45$**，并生成连贯的故事文本。
- [ ] **完成 Section 7 实验**：
  1. 学习率扫描（找到最优 LR 与发散临界点）；
  2. Batch Size 扫描；
  3. 架构消融实验：移除 RMSNorm、Pre-Norm 改 Post-Norm、移除 RoPE（NoPE）、SwiGLU 改 SiLU；
  4. 在 `owt_train.txt` 上用相同架构训练 OpenWebText 语言模型。
