# Stanford CS336 从零手搓大模型：全流程对照学习手册 (Study Roadmap)

这份文档专为你的本地学习工程定制。你可以直接在 IDE（PyCharm / VS Code）中打开本文件作为总纲，并且**每个作业目录下都已为你生成了更加深入、包含数学公式、目录结构规划、核心实现细节、避坑指南与单测命令的专属通关手册 `STUDY_GUIDE.md`**：

* 📘 **Assignment 1 详细手册**：[assignment1-basics/STUDY_GUIDE.md](./assignment1-basics/STUDY_GUIDE.md)
* 📗 **Assignment 2 详细手册**：[assignment2-systems/STUDY_GUIDE.md](./assignment2-systems/STUDY_GUIDE.md)
* 📙 **Assignment 3 详细手册**：[assignment3-scaling/STUDY_GUIDE.md](./assignment3-scaling/STUDY_GUIDE.md)
* 📒 **Assignment 4 详细手册**：[assignment4-data/STUDY_GUIDE.md](./assignment4-data/STUDY_GUIDE.md)
* 📕 **Assignment 5 详细手册**：[assignment5-alignment/STUDY_GUIDE.md](./assignment5-alignment/STUDY_GUIDE.md)

每完成一个小节的 **看课 $\to$ 读 PDF $\to$ 写代码 $\to$ 跑测试 $\to$ 做实验**，就在对应的 `[ ]` 中打勾 `[x]`。

---

## 📌 全局学习方法论（怎么最高效地刷完 CS336？）

每个模块都严格遵循以下 **4 步闭环**：
1. **第一步（看课 & 读讲义代码）**：先看对应的视频 / 阅读 `lectures/lecture_XX.py` 或 `lecture_XX.pdf`。注意不要只听概念，`.py` 讲义里本身就是可执行的推导代码。
2. **第二步（精读 Assignment PDF 指定小节）**：**不要一口气把几十页 PDF 从头读到尾！** 每次只读当前子任务对应的 2~4 页，看清数学公式、输入输出张量形状（Shape）和边界要求。
3. **第三步（写核心实现 $\to$ 连 `adapters.py` $\to$ 跑 `pytest`）**：
   * 在各作业的包目录（如 `cs336_basics/`）中新建模块编写实现；
   * 在 `tests/adapters.py` 对应的桩函数中导入并调用你的实现；
   * 运行对应的 `uv run pytest tests/test_xxx.py`，**绿灯通过后再进入下一个函数**。
4. **第四步（跑实验 & 记录思考题 Deliverable）**：利用你本机的 **RTX 5090 D (32GB VRAM)** 跑真实训练或剖析实验，回答 PDF 中标有 **Deliverable** 的简答题。

---

## 阶段一：Assignment 1 — 语言模型基石 (`assignment1-basics`)

* **工作目录**：`cd /home/yu/projects/learning/cs336/assignment1-basics`
* **作业文档**：[cs336_assignment1_basics.pdf](./assignment1-basics/cs336_assignment1_basics.pdf)
* **适配器文件**：[assignment1-basics/tests/adapters.py](./assignment1-basics/tests/adapters.py)

---

### 模块 1.1：字符编码与 BPE 分词器训练 (Byte-Pair Encoding)

#### 📺 1. 看课与讲义到什么程度？
- [ ] 观看 **Lecture 1 (Overview & Tokenization)** 视频前半部分（历史背景可快进，重点看 Tokenization 部分）。
- [ ] 打开并阅读 [lectures/lecture_01.py](./lectures/lecture_01.py)：
  - 看到 `tokenization()` 函数部分，理解 Unicode 码点（`ord`/`chr`）、UTF-8 字节编码，以及为什么直接用 UTF-8 字节序列会导致序列过长，从而引出 BPE 算法。

#### 📖 2. 读 Assignment 1 PDF 到什么程度？
- [ ] 阅读 [cs336_assignment1_basics.pdf](./assignment1-basics/cs336_assignment1_basics.pdf) 的 **Section 1 (Overview)** 与 **Section 2.1 ~ 2.5 (第 1 页 ~ 第 9 页)**：
  - **2.1 Unicode Standard & 2.2 Unicode Encodings**：回答 PDF 中的基础思考题（`chr(0)`、UTF-8 vs UTF-16/32 的原因、直接逐字节解码的错误例子）。
  - **2.3 Subword Tokenization & 2.4 BPE Tokenizer Training**：重点看清三件事：
    1. 预分词正则 `PAT` 的作用（防止跨标点/跨词合并）；
    2. `special_tokens`（如 `<|endoftext|>`）必须在正则预分词**之前**先切开，绝不能参与内部字节合并；
    3. 当两个 pair 出现频次相同时，**取字典序更大（lexicographically greater）** 的 pair。
  - **2.5 Experimenting with BPE Tokenizer Training**：看清 TinyStories (`vocab_size=10000`) 和 OpenWebText (`vocab_size=32000`) 的训练要求。

#### 💻 3. 动手做什么事情？
- [x] **实现基础版 `run_train_bpe`**：在 [adapters.py](./assignment1-basics/tests/adapters.py) 中实现 `run_train_bpe`，跑通单元测试：
  ```bash
  uv run pytest tests/test_train_bpe.py
  ```
- [ ] **升级多进程预分词 + 最大堆优化**：对照参考实现 [bpe_optimized_ref.py](./assignment1-basics/tests/yu/bpe_optimized_ref.py)，理解：
  1. 如何用 `find_chunk_boundaries` 按 `<|endoftext|>` 对齐切分大文件并用 `multiprocessing.Pool` 并行统计词频；
  2. 如何用 `heapq` + `_PairItem`（惰性删除 Lazy Deletion）+ `changed_pairs` 批量入堆替代每轮全表 `max()`。
- [ ] **完成 Problem (`train_bpe_tinystories`) & (`train_bpe_expts_owt`)**：
  - 在 [tests/yu/test.ipynb](./assignment1-basics/tests/yu/test.ipynb) 或独立脚本中，对 `data/TinyStoriesV2-GPT4-train.txt` 训练 `vocab_size=10000` 的词表，将 `vocab` 和 `merges` 序列化保存到磁盘。
  - 统计训练耗时与内存峰值，找出词表中最长的 token（看看它对应什么英文单词）。
  - 用 `cProfile` 剖析训练耗时占比最高的部分。
  - 对 `data/owt_train.txt` (12GB) 训练 `vocab_size=32000` 的词表并保存，对比 TinyStories 与 OpenWebText 词表的差异。

---

### 模块 1.2：BPE 分词器推理（Encode / Decode / 流式编码）

#### 📖 1. 读 Assignment 1 PDF 到什么程度？
- [ ] 阅读 [cs336_assignment1_basics.pdf](./assignment1-basics/cs336_assignment1_basics.pdf) 的 **Section 2.6 ~ 2.7 (第 10 页 ~ 第 13 页)**：
  - 看清 `Tokenizer` 类的 5 个核心方法签名：`__init__`、`from_files`、`encode(text)`、`encode_iterable(iterable)`、`decode(ids)`。
  - **重点注意**：
    1. `encode` 时，对于每个预分词出来的普通词块，要按训练时的 `merges` **先后顺序（即优先级 rank 最小优先）** 依次合并，而不是按词频合并！建议在 `__init__` 中构建 `merge_ranks: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(merges)}`。
    2. 当存在重叠的 `special_tokens`（如 `["<|endoftext|>", "<|endoftext|><|endoftext|>"]`）时，正则切分必须按**长度降序**排列，且用捕获括号 `(...)` 保留切分出来的特殊 token，直接映射为其对应的整数 ID。
    3. `decode` 时，拼接所有 token 的 `bytes` 后调用 `.decode("utf-8", errors="replace")` 处理非法字节。

#### 💻 2. 动手做什么事情？
- [ ] 在 `cs336_basics/` 下创建 `tokenizer.py`，实现 `Tokenizer` 类，并在 [adapters.py](./assignment1-basics/tests/adapters.py) 中完成 `get_tokenizer`。
- [ ] 运行并跑通所有分词器单测：
  ```bash
  uv run pytest tests/test_tokenizer.py
  ```
- [ ] **完成 Section 2.7 实验 (`tokenizer_experiments`)**：
  - 采样 10 篇 TinyStories 和 10 篇 OpenWebText 文档，计算压缩率（`bytes / token`）。
  - 测算你的分词器吞吐量（`bytes / second`），估算 tokenize 整个 Pile 数据集（825GB）需要多久。
  - 使用 `encode_iterable` 将 `TinyStories` 和 `owt` 的 train/valid 文本编码为 `uint16` 的 NumPy 数组（`np.uint16`）并保存为 `.npy` 或 `.bin` 文件，供后续模型训练使用。

---

### 模块 1.3：从零搭建现代 Transformer 架构 (Pre-Norm + RMSNorm + SwiGLU + RoPE)

#### 📺 1. 看课与讲义到什么程度？
- [ ] 观看 **Lecture 2 (PyTorch, Resource Accounting)** 视频，并阅读 [lectures/lecture_02.py](./lectures/lecture_02.py)：
  - 重点掌握张量内存布局、`einops`（`einsum` / `rearrange`）、矩阵乘法 FLOPs 计算公式（形状 `[B, T, D]` 乘 `[D, K]` 的 FLOPs 为 $2 \cdot B \cdot T \cdot D \cdot K$）。
- [ ] 观看 **Lecture 3 (Architecture)** 视频，并阅读课件 [lectures/lecture_03.pdf](./lectures/lecture_03.pdf)：
  - 看懂 LLaMA 风格现代架构与 2017 原版 Transformer 的 4 大区别：
    1. **Pre-Norm**（Norm 在残差分支入口，主干无 Norm）；
    2. **RMSNorm** 替代 LayerNorm（去掉均值中心化与偏置项）；
    3. **SwiGLU** 替代 ReLU FFN（3 个矩阵 $W_1, W_2, W_3$，中间维度 $d_{ff} \approx \frac{8}{3} d_{\text{model}}$ 且向 64 倍数对齐）；
    4. **RoPE (Rotary Position Embeddings)** 旋转位置编码替代绝对位置编码。
- [ ] *(选看拓展)* 翻阅 [lectures/lecture_04.pdf](./lectures/lecture_04.pdf) 了解 MoE（混合专家模型）架构。

#### 📖 2. 读 Assignment 1 PDF 到什么程度？
- [ ] 阅读 [cs336_assignment1_basics.pdf](./assignment1-basics/cs336_assignment1_basics.pdf) 的 **Section 3.1 ~ 3.7 (第 13 页 ~ 第 26 页)**：
  - 每读完一个小组件（3.4.1 Linear $\to$ 3.4.2 Embedding $\to$ 3.5.1 RMSNorm $\to$ 3.5.2 SwiGLU $\to$ 3.5.3 RoPE $\to$ 3.5.4 Scaled Dot-Product Attention & Multi-Head Attention $\to$ 3.6 Transformer Block $\to$ 3.7 Full Transformer LM & 算力/显存核算），立刻去写对应的模块！

#### 💻 3. 动手做什么事情？（按自底向上顺序逐个攻破）
- [ ] **Step 1：实现 `Linear` 与 `Embedding`**（Section 3.4.1 & 3.4.2）
  - 注意：`Linear` 权重形状为 `(d_out, d_in)`，不含 bias，使用 `nn.init.trunc_normal_` 初始化。
  - 对接 [adapters.py](./assignment1-basics/tests/adapters.py) 中的 `run_linear` 和 `run_embedding`。
  - 验证：`uv run pytest tests/test_model.py -k "test_linear or test_embedding"`
- [ ] **Step 2：实现 `RMSNorm`**（Section 3.5.1）
  - 注意：**必须先将输入 `x` 转为 `torch.float32`** 计算均方根，算完后再转回原 `dtype`，防止 FP16/BF16 下平方溢出！
  - 对接 `run_rmsnorm`，验证：`uv run pytest tests/test_model.py -k "test_rmsnorm"`
- [ ] **Step 3：实现 `SiLU` 与 `SwiGLU` 前馈网络**（Section 3.5.2）
  - $\text{SiLU}(x) = x \cdot \sigma(x)$，$\text{SwiGLU}(x) = W_2(\text{SiLU}(W_1 x) \odot W_3 x)$。
  - 对接 `run_silu` 和 `run_swiglu`，验证：`uv run pytest tests/test_model.py -k "test_silu or test_swiglu"`
- [ ] **Step 4：实现旋转位置编码 `RotaryPositionalEmbedding (RoPE)`**（Section 3.5.3）
  - 预计算 $\cos$ 和 $\sin$ 缓存并注册为 `register_buffer(..., persistent=False)`，将最后维按相邻两元素 `(x[..., 0::2], x[..., 1::2])` 分组旋转。
  - 对接 `run_rope`，验证：`uv run pytest tests/test_model.py -k "test_rope"`
- [ ] **Step 5：实现数值稳定 `softmax` 与 `scaled_dot_product_attention`**（Section 3.5.4）
  - `softmax`：减去目标维度上的最大值 `x - x.max(dim=dim, keepdim=True).values` 后再 `exp`。
  - `attention`：计算 $\frac{Q K^T}{\sqrt{d_k}}$，将 `mask == False` 的位置填充为 `-inf`（`torch.where(mask, scores, float("-inf"))`），过 `softmax` 后乘 $V$。
  - 对接 `run_softmax` 与 `run_scaled_dot_product_attention`，验证：`uv run pytest tests/test_nn_utils.py -k "test_softmax"` 及 `uv run pytest tests/test_model.py -k "test_scaled_dot_product_attention"`
- [ ] **Step 6：实现因果多头自注意力 `CausalMultiHeadSelfAttention`**（Section 3.5.5）
  - 支持带 RoPE（只对 $Q$ 和 $K$ 施加 RoPE，$V$ 不加）与不带 RoPE，构造下三角因果掩码 `torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))`。
  - 对接 `run_multihead_self_attention` 和 `run_multihead_self_attention_with_rope`，验证：`uv run pytest tests/test_model.py -k "multihead"`
- [ ] **Step 7：组装 `TransformerBlock` 与完整 `TransformerLM`**（Section 3.6 & 3.7）
  - 对接 `run_transformer_block` 和 `run_transformer_lm`，跑通整个模型测试文件：
    ```bash
    uv run pytest tests/test_model.py
    ```
- [ ] **Step 8：完成 Section 3.7 资源核算题 (`transformer_accounting`)**
  - 按 GPT-2 XL 参数配置手算（或写个 10 行小脚本辅助核算）参数量、前向 FLOPs 各层占比，以及不同模型尺寸下的峰值显存。

---

### 模块 1.4：交叉熵、AdamW 优化器、数据加载与完整训练循环

#### 📺 1. 看课与讲义到什么程度？
- [ ] 回看 **Lecture 2** 的后半部分（[lectures/lecture_02.py](./lectures/lecture_02.py) 中的 `training_loop`、`adamw` 与显存分析：模型参数 + 梯度 + AdamW 的 `m` 和 `v` 一共占多少字节/参数）。

#### 📖 2. 读 Assignment 1 PDF 到什么程度？
- [ ] 阅读 [cs336_assignment1_basics.pdf](./assignment1-basics/cs336_assignment1_basics.pdf) 的 **Section 4 ~ Section 7 (第 26 页 ~ 第 40 页)**：
  - **Section 4**：数值稳定 Cross-Entropy（LogSumExp 技巧，消去一次 `softmax` + `log`）、SGD 实验、手写 `AdamW`（带偏置修正 $\alpha_t = \alpha \frac{\sqrt{1-\beta_2^t}}{1-\beta_1^t}$ 和解耦权重衰减）、余弦退火学习率调度（带线性 Warmup）、全局梯度 L2 范数裁剪。
  - **Section 5**：`get_batch`（支持 `np.memmap` 按需从磁盘切片）、`save_checkpoint` / `load_checkpoint`、完整训练脚本组装。
  - **Section 6 & 7**：解码采样（Temperature Scaling + Top-p Nucleus Sampling）与训练实验（学习率搜索、Batch Size 实验、架构消融实验、OpenWebText 训练）。

#### 💻 3. 动手做什么事情？
- [ ] **实现 `cross_entropy` 与 `gradient_clipping`**：
  - 对接 `run_cross_entropy` 和 `run_gradient_clipping`，验证：
    ```bash
    uv run pytest tests/test_nn_utils.py
    ```
- [ ] **实现 `AdamW` 与 `get_lr_cosine_schedule`**：
  - 对接 `get_adamw_cls` 和 `run_get_lr_cosine_schedule`，验证：
    ```bash
    uv run pytest tests/test_optimizer.py
    ```
- [ ] **实现 `get_batch` 与 Checkpoint 序列化**：
  - 对接 `run_get_batch`、`run_save_checkpoint`、`run_load_checkpoint`，验证：
    ```bash
    uv run pytest tests/test_data.py tests/test_serialization.py
    ```
- [ ] **确认 Assignment 1 全量单元测试 100% 绿灯**：
  ```bash
  uv run pytest
  ```
- [ ] **编写训练主脚本 `cs336_basics/train.py` 与生成脚本 `decode.py`**：
  - 在 RTX 5090 D 上训练 TinyStories 模型（目标验证集 Loss $\le 1.45$），用 `decode.py` 生成故事。
  - 完成消融实验（去掉 RMSNorm、把 Pre-Norm 改为 Post-Norm、去掉 RoPE 改 NoPE、把 SwiGLU 改为 SiLU）并对比 Loss 曲线。

---

## 阶段二：Assignment 2 — GPU 系统性能、Triton 内核与分布式训练 (`assignment2-systems`)

* **工作目录**：`cd /home/yu/projects/learning/cs336/assignment2-systems`
* **作业文档**：[cs336_assignment2_systems.pdf](./assignment2-systems/cs336_assignment2_systems.pdf)
* **适配器文件**：[assignment2-systems/tests/adapters.py](./assignment2-systems/tests/adapters.py)

---

### 模块 2.1：GPU 性能剖析、混合精度与手写 FlashAttention-2 (PyTorch + Triton)

#### 📺 1. 看课与讲义到什么程度？
- [ ] 观看 **Lecture 5 (GPUs)** 视频，阅读课件 [lectures/lecture_05.pdf](./lectures/lecture_05.pdf)：
  - 彻底搞懂 GPU 内存层级（寄存器 / SRAM L1 Shared Memory vs HBM 显存）、带宽瓶颈（Memory-bound vs Compute-bound）、为什么朴素 Attention 要频繁读写 $N \times N$ 矩阵，以及 **FlashAttention 在线 Softmax 分块（Tiling）** 的数学推导。
- [ ] 观看 **Lecture 6 (Kernels & Triton)** 视频，精读可执行讲义 [lectures/lecture_06.py](./lectures/lecture_06.py)：
  - 重点看 `benchmarking_and_profiling()`、`triton_gelu_example()`、`triton_softmax_example()`，掌握 Triton 的 `tl.program_id`、块指针 `tl.make_block_ptr`、掩码加载与规约。

#### 📖 2. 读 Assignment 2 PDF 到什么程度？
- [ ] 阅读 [cs336_assignment2_systems.pdf](./assignment2-systems/cs336_assignment2_systems.pdf) 的 **Section 1 (Profiling and Benchmarking)** 与 **Section 2 (FlashAttention-2)**：
  - 仔细对照 Section 2 中给出的 FlashAttention-2 前向（Forward）分块递推公式（维护块内最大值 $m_i$、指数和 $l_i$、累加输出 $O_i$ 以及保存 $L_i = m_i + \log l_i$）和反向（Backward）重计算公式。

#### 💻 3. 动手做什么事情？
- [ ] **完成 Section 1 性能剖析实验**：
  - 编写基准计时脚本（带 warmup + `torch.cuda.synchronize()`），对比 FP32 vs `torch.autocast("cuda", dtype=torch.bfloat16)` vs `torch.compile` 的前反向耗时与显存占用。
- [ ] **实现 PyTorch 分块版 FlashAttention-2**：
  - 继承 `torch.autograd.Function` 实现前向与反向分块逻辑，对接 [adapters.py](./assignment2-systems/tests/adapters.py) 中的 `get_flashattention_autograd_function_pytorch`。
  - 验证：`uv run pytest tests/test_attention.py -k "pytorch"`
- [ ] **实现 Triton 内核版 FlashAttention-2**：
  - 用 `@triton.jit` 编写前向 Kernel（支持因果掩码 `is_causal`），配合反向传播对接 `get_flashattention_autograd_function_triton`。
  - 验证：
    ```bash
    uv run pytest tests/test_attention.py
    ```

---

### 模块 2.2：分布式数据并行 (DDP)、优化器状态分片 (ZeRO-1) 与 FSDP

#### 📺 1. 看课与讲义到什么程度？
- [ ] 观看 **Lecture 7 (Parallelism I)** 与 **Lecture 8 (Parallelism II)**，阅读 [lectures/lecture_07.py](./lectures/lecture_07.py) 和 [lectures/lecture_08.pdf](./lectures/lecture_08.pdf)：
  - 重点看 [lecture_07.py](./lectures/lecture_07.py) 中的集合通信原语（`all_reduce`、`reduce_scatter`、`all_gather`）以及 `data_parallelism()` / ZeRO Stage 1, 2, 3 的显存拆分推导。

#### 📖 2. 读 Assignment 2 PDF 到什么程度？
- [ ] 阅读 [cs336_assignment2_systems.pdf](./assignment2-systems/cs336_assignment2_systems.pdf) 的 **Section 3 (Distributed Data Parallel)** 与 **Section 4 (Optimizer State Sharding / FSDP)**。

#### 💻 3. 动手做什么事情？
- [ ] **实现 DDP（含梯度分桶与异步通信重叠）**：
  - 利用 `register_post_accumulate_grad_hook` 在反向传播过程中异步触发 `dist.all_reduce(..., async_op=True)`，对接 `get_ddp` 与 `ddp_on_after_backward`。
  - 验证：`uv run pytest tests/test_ddp.py`
- [ ] **实现优化器状态分片 (`ShardedOptimizer` / ZeRO-1) 与 FSDP**：
  - 每个 rank 只维护属于自己分片参数的优化器状态，更新后广播（或 `all_gather`）同步参数，对接 `get_sharded_optimizer` 及 `get_fsdp` 相关接口。
  - 验证：
    ```bash
    uv run pytest tests/test_sharded_optimizer.py tests/test_fsdp.py
    ```

---

## 阶段三：Assignment 3 — 缩放定律 (Scaling Laws) 与推理 (`assignment3-scaling`)

* **工作目录**：`cd /home/yu/projects/learning/cs336/assignment3-scaling`
* **作业文档**：[cs336_assignment3_scaling.pdf](./assignment3-scaling/cs336_assignment3_scaling.pdf)

#### 📺 1. 看课与讲义到什么程度？
- [ ] 观看 **Lecture 9 & Lecture 11 (Scaling Laws)**，阅读 [lectures/lecture_09.pdf](./lectures/lecture_09.pdf) 与 [lectures/lecture_11.pdf](./lectures/lecture_11.pdf)：
  - 理解 Chinchilla 的三种方法（Approach 1: 固定模型大小看不同 Token 数；Approach 2: IsoFLOPs 曲线拟合法；Approach 3: 参数化拟合 $L(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$），以及 $\mu$P 跨尺寸超参数迁移。
- [ ] 观看 **Lecture 10 (Inference)**，阅读 [lectures/lecture_10.py](./lectures/lecture_10.py)：
  - 理解 Prefill（算力受限）与 Decode（访存受限）、KV Cache 显存计算、GQA/MLA 如何压缩 KV Cache。

#### 📖 2. 读 Assignment 3 PDF 并动手实践
- [ ] 阅读 [cs336_assignment3_scaling.pdf](./assignment3-scaling/cs336_assignment3_scaling.pdf) 全文，查看 `assignment3-scaling/data/` 与 `examples/` 中的实验记录。
- [ ] 跑通单元测试：`uv run pytest`
- [ ] 编写曲线拟合脚本（使用 `scipy.optimize.curve_fit` 或二次抛物线拟合 IsoFLOPs 最低点），在给定 FLOPs 预算约束下，拟合出 $N_{\text{opt}}(C) \propto C^a$ 与 $D_{\text{opt}}(C) \propto C^b$，给出目标大算力下的最优模型配置预测。

---

## 阶段四：Assignment 4 — 预训练数据清洗与去重流水线 (`assignment4-data`)

* **工作目录**：`cd /home/yu/projects/learning/cs336/assignment4-data`
* **作业文档**：[cs336_assignment4_data.pdf](./assignment4-data/cs336_assignment4_data.pdf)
* **适配器文件**：[assignment4-data/tests/adapters.py](./assignment4-data/tests/adapters.py)

#### 📺 1. 看课与讲义到什么程度？
- [ ] 观看 **Lecture 12 (Evaluation)**，阅读 [lectures/lecture_12.py](./lectures/lecture_12.py)。
- [ ] 观看 **Lecture 13 (Data I)** 与 **Lecture 14 (Data II)**，精读 [lectures/lecture_13.py](./lectures/lecture_13.py) 和 [lectures/lecture_14.py](./lectures/lecture_14.py)：
  - 重点看 [lecture_14.py](./lectures/lecture_14.py) 中的 `transformation()`（HTML 转文本）、`filtering()`（fastText 分类器、启发式规则）和 `deduplication()`（Jaccard 相似度、MinHash 签名矩阵、LSH 分桶原理）。

#### 📖 2. 读 Assignment 4 PDF 并逐个模块通关
- [ ] **Step 1：HTML 文本提取（Section 2.1）**
  - 用 `resiliparse` 检测编码并提取网页正文，对接 `run_extract_text_from_html_bytes`。
  - 验证：`uv run pytest tests/test_extract.py`
- [ ] **Step 2：语种识别（Section 2.2）**
  - 用 `fasttext` 语言识别模型预测语言标签与置信度，对接 `run_identify_language`。
  - 验证：`uv run pytest tests/test_langid.py`
- [ ] **Step 3：个人隐私信息 PII 脱敏（Section 2.3）**
  - 用正则表达式将邮箱、电话号码、IPv4 地址替换为指定掩码并统计替换次数，对接 `run_mask_emails`、`run_mask_phone_numbers`、`run_mask_ips`。
  - 验证：`uv run pytest tests/test_pii.py`
- [ ] **Step 4：有害内容识别与 Gopher 启发式质量过滤（Section 2.4 & 2.5）**
  - 实现 NSFW / Toxic 分类器调用（`run_classify_nsfw`、`run_classify_toxic_speech`）与 Gopher 规则过滤（词数范围、平均词长、省略号行占比、含字母词占比：`run_gopher_quality_filter`）。
  - 验证：`uv run pytest tests/test_toxicity.py tests/test_gopher.py`（注：检查 `tests/` 下的对应测试名）
- [ ] **Step 5：训练并运行语料质量分类器（Section 2.6）**
  - 对接 `run_classify_quality`，验证：`uv run pytest tests/test_quality.py`
- [ ] **Step 6：精确行级去重与 MinHash + LSH 模糊文档去重（Section 3.1 & 3.2）**
  - 实现两遍扫描的精确行级去重 `run_exact_line_deduplication`。
  - 实现 n-gram MinHash 签名 + LSH 分带（Bands/Rows）候选对发现 + 真实 Jaccard 校验聚类去重 `run_minhash_deduplication`。
  - 验证：`uv run pytest tests/test_deduplication.py`

---

## 阶段五：Assignment 5 — 后训练对齐、SFT、DPO 与推理强化学习 GRPO (`assignment5-alignment`)

* **工作目录**：`cd /home/yu/projects/learning/cs336/assignment5-alignment`
* **作业文档**：
  - 主文档（数学推理与 GRPO）：[cs336_spring2026_assignment5_alignment.pdf](./assignment5-alignment/cs336_spring2026_assignment5_alignment.pdf)
  - 补充文档（安全对齐与 DPO）：[cs336_spring2026_assignment5_supplement_safety_rlhf.pdf](./assignment5-alignment/cs336_spring2026_assignment5_supplement_safety_rlhf.pdf)
* **适配器文件**：[assignment5-alignment/tests/adapters.py](./assignment5-alignment/tests/adapters.py)

#### 📺 1. 看课与讲义到什么程度？
- [ ] 观看 **Lecture 15 (Alignment I: SFT & RLHF)** 与 **Lecture 16 (Alignment II: DPO & GRPO)**，阅读 [lectures/lecture_15.pdf](./lectures/lecture_15.pdf) 和 [lectures/lecture_16.pdf](./lectures/lecture_16.pdf)：
  - 重点理解：
    1. SFT 只对 Response 部分计算 Loss（Prompt 部分 Mask 掉）；
    2. DPO 如何通过闭式解把 Reward Model 消掉，直接用 Policy 和 Reference Policy 的对数概率比做二分类损失；
    3. **GRPO (Group Relative Policy Optimization，DeepSeek-R1 核心算法)**：如何去掉庞大的 Critic (Value) 网络，通过对同一个问题采样一组（Group $G$ 个）回答，用组内奖励的均值和标准差计算相对优势 $A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$。

#### 📖 2. 读 Assignment 5 PDF 并逐个模块通关
- [ ] **Step 1：评估解析器与零样本基线（主 PDF Section 2 & 补充 PDF）**
  - 实现 GSM8K 与 MMLU 答案解析器 `run_parse_gsm8k_response`、`run_parse_mmlu_response`。
  - 验证：`uv run pytest tests/test_metrics.py`
- [ ] **Step 2：SFT 数据打包与训练组件（主 PDF Section 3 & 补充 PDF）**
  - 实现 Prompt + Output 拼接编码与 `response_mask` 构造（`run_tokenize_prompt_and_output`）、对数概率提取与 Token 级熵计算（`run_get_response_log_probs`）、打包数据集（`get_packed_sft_dataset`、`run_iterate_batches`）。
  - 验证：`uv run pytest tests/test_data.py`
- [ ] **Step 3：GRPO 强化学习核心流水线（主 PDF Section 4）**
  - 依次实现并在 [adapters.py](./assignment5-alignment/tests/adapters.py) 中对接：
    1. `run_compute_rollout_rewards`（调用数学答案校验器打分）
    2. `run_compute_group_normalized_rewards`（组内标准化 Advantage）
    3. `run_compute_policy_gradient_loss`（支持朴素策略梯度、带/不带标准差归一化的 GRPO-Clip 损失）
    4. `run_aggregate_loss_across_microbatch` 與 `run_grpo_train_step`
  - 验证：`uv run pytest tests/test_grpo.py`
  - 在你的 RTX 5090 D 上运行 GRPO 训练实验，观察数学推理正确率随训练步数的爬升！
- [ ] **Step 4：DPO 直接偏好优化（补充 PDF）**
  - 实现单样本 DPO 损失函数 `run_compute_per_instance_dpo_loss`：
    $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)$$
  - 验证：`uv run pytest tests/test_dpo.py`

---

## 阶段六：拓展收官 — 多模态大模型 (Lecture 17)
- [ ] 观看 **Lecture 17 (Multimodal Models)**，阅读 [lectures/lecture_17.py](./lectures/lecture_17.py)，了解视觉编码器（ViT Patch Tokenization）与连续/离散多模态 Token 如何接入你在前面亲手实现的 Transformer 主干中。
