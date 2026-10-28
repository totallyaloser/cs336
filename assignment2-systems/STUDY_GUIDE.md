# Assignment 2: Systems — 详细通关实战指南

> **本目录目标**：深入 GPU 底层硬件与分布式系统，在你的 **RTX 5090 D (32GB)** 上完成 **模型性能剖析 (Profiling) + 手写 FlashAttention-2 (PyTorch & Triton 内核) + 异步通信重叠 DDP + 优化器分片 (ZeRO-1) 与 FSDP**。
> *注：本目录已内置官方参考基线 `cs336-basics/`，无需依赖你 Assignment 1 的代码即可直接开始！*

---

## 📁 推荐你在 `cs336_systems/` 下建立的工程结构

```text
assignment2-systems/
├── cs336_systems/
│   ├── __init__.py
│   ├── benchmark.py            # [阶段 1] Warmup + CUDA 同步计时、混合精度、PyTorch Profiler 脚本
│   ├── flash_attention.py      # [阶段 2] FlashAttention-2 的 PyTorch 分块版与 Triton 内核版
│   ├── ddp.py                  # [阶段 3] 异步梯度通信重叠的 DDP 容器
│   ├── sharded_optimizer.py    # [阶段 4] ZeRO-1 优化器状态分片容器
│   └── fsdp.py                 # [阶段 4] FSDP 完全分片数据并行容器
└── tests/
    └── adapters.py             # 连接上述实现与 pytest 单元测试
```

---

## 第一阶段：GPU 基准测试、性能剖析与混合精度 (`cs336_systems/benchmark.py`)

### 1. 学习准备
- [ ] **看课**：
  - 观看 **Lecture 5 (GPU Architecture)**，阅读 [../lectures/lecture_05.pdf](../lectures/lecture_05.pdf)：理解 SM（流多处理器）、HBM（高带宽显存）与 SRAM（片上共享内存/寄存器）的速度差、Arithmetic Intensity（算术强度 = FLOPs / Bytes）。
  - 观看 **Lecture 6 (Profiling & Kernels)** 前半部分，阅读 [../lectures/lecture_06.py](../lectures/lecture_06.py) 中的 `benchmarking_and_profiling()` 和 `naive_vs_builtin_vs_compiled_gelu()`。
- [ ] **读 PDF**：[cs336_assignment2_systems.pdf](./cs336_assignment2_systems.pdf) **Section 1.1 ~ 1.3**。

### 2. 动手实践清单
- [ ] **编写精确的 GPU 计时脚本 (`benchmark.py`)**：
  - **必做避坑点**：CUDA 核函数启动是**异步**的！每次计时前必须先跑 $w$ 步（如 5 步）Warmup，并在 `timeit.default_timer()` 记录前后调用 **`torch.cuda.synchronize()`**！
  - 分别测量 Small / Medium / Large / XL 模型尺寸下，纯前向（Forward）和前向+反向（Forward + Backward）的平均耗时与标准差。
- [ ] **使用 PyTorch Profiler / Nsight Systems 分析瓶颈**：
  - 找出前向与反向中耗时占比最高的 CUDA 算子（矩阵乘法 `gemm` vs 逐元素算子 vs Softmax）。
- [ ] **混合精度（Mixed Precision `bfloat16`）与算子编译（`torch.compile`）实验**：
  - 理解为什么 `RMSNorm` 和累加需要在 `float32` 下进行，而线性层矩阵乘法 (`matmul`) 用 `bfloat16` 能获得数倍 Tensor Core 加速。
  - 使用 `torch.cuda.memory._record_memory_history()` 导出显存快照，拖入 `https://pytorch.org/memory_viz` 观察前向激活值、反向梯度与优化器状态的显存波形图。

---

## 第二阶段：手写 FlashAttention-2 (`cs336_systems/flash_attention.py`)

### 1. 学习准备
- [ ] **看课**：精读 [../lectures/lecture_06.py](../lectures/lecture_06.py) 中的 `triton_introduction()`、`triton_softmax_example()` 和 `triton_row_sum_example()`。
- [ ] **读 PDF**：[cs336_assignment2_systems.pdf](./cs336_assignment2_systems.pdf) **Section 2 (FlashAttention-2)**，逐行看懂算法伪代码！

### 2. 核心数学原理与避坑指南

#### 任务 2.1：纯 PyTorch 分块版 FlashAttention-2 (`get_flashattention_autograd_function_pytorch`)
先不用写 Triton，用纯 PyTorch 张量循环把分块在线 Softmax（Online Softmax Tiling）逻辑跑通！
- **输入形状**：`Q, K, V` 均为 `(batch_size, seq_len, d_head)`，以及布尔值 `is_causal: bool = False`。
- **前向分块递推公式**（将 $Q$ 按行切分为块大小 $B_q$，将 $K, V$ 按行切分为块大小 $B_k$）：
  对第 $i$ 个 Query 块 $Q_i \in \mathbb{R}^{B_q \times d}$，初始化运行最大值 $m_i^{(0)} = -\infty$，分母累加和 $l_i^{(0)} = 0$，分子输出 $O_i^{(0)} = \mathbf{0}$。
  内层遍历第 $j$ 个 Key/Value 块 $(K_j, V_j)$：
  1. 计算块注意力分数：$S_i^{(j)} = \frac{Q_i K_j^T}{\sqrt{d}}$（若 `is_causal=True`，对行下标 $<$ 列下标的位置填入 `-1e6`）。
  2. 更新行最大值：$m_i^{(j)} = \max\big(m_i^{(j-1)}, \text{rowmax}(S_i^{(j)})\big)$。
  3. 计算未归一化概率：$\tilde{P}_i^{(j)} = \exp(S_i^{(j)} - m_i^{(j)})$。
  4. 缩放修正分母：$l_i^{(j)} = \exp(m_i^{(j-1)} - m_i^{(j)}) l_i^{(j-1)} + \text{rowsum}(\tilde{P}_i^{(j)})$。
  5. 缩放修正并累加输出：$O_i^{(j)} = \text{diag}\big(\exp(m_i^{(j-1)} - m_i^{(j)})\big) O_i^{(j-1)} + \tilde{P}_i^{(j)} V_j$。
  内层循环结束后：
  - 最终输出块：$O_i = \text{diag}(l_i)^{-1} O_i^{(T_k)}$
  - 对数配分函数（LogSumExp）：$L_i = m_i^{(T_k)} + \log(l_i^{(T_k)})$
- ⚠️ **单测强制要求（见 [tests/test_attention.py](./tests/test_attention.py#L48-L54)）**：
  在 `forward` 结束时，**必须调用 `ctx.save_for_backward(L, Q, K, V, O)`**，其中保存的 `L` 形状**必须恰好为 `(batch_size, n_queries)`**（并且只能有 `L` 这一个张量是该形状）！
- **反向传播 (`backward(ctx, dO)`)**：
  - 预计算行内点积 $D = \text{rowsum}(O \odot dO)$（形状 `(batch_size, n_queries)`）。
  - 利用保存的 $L$ 重算 $P_{ij} = \exp(S_{ij} - L_i)$，$dV_j += P_{ij}^T dO_i$，$dP_{ij} = dO_i V_j^T$，$dS_{ij} = P_{ij} \odot (dP_{ij} - D_i)$，$dQ_i += \frac{1}{\sqrt{d}} dS_{ij} K_j$，$dK_j += \frac{1}{\sqrt{d}} dS_{ij}^T Q_i$。
- [ ] **验收 PyTorch 版前反向单测**：
  ```bash
  uv run pytest tests/test_attention.py -k "pytorch"
  ```

#### 任务 2.2：Triton 内核版 FlashAttention-2 (`get_flashattention_autograd_function_triton`)
- **网格设计 (Launch Grid)**：`grid = (triton.cdiv(N_QUERIES, Q_TILE_SIZE), batch_size)`，每个 Triton Program 实例负责 1 个 Batch 样本中的 1 个 Query Tile！
- **指针构造**：使用 `tl.make_block_ptr` 为 `Q`, `K`, `V`, `O`, `L` 构造块指针，在 K/V 循环末尾用 `K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))` 推进指针。
- **精度避坑点**：片上累加器 `acc`、`m_i`、`l_i` **必须使用 `tl.float32`**，矩阵乘法使用 `tl.dot(q, tl.trans(k))`；写入输出前再将 `acc` 转换回 `O_block_ptr.type.element_ty`。
- **因果掩码 (`is_causal: tl.constexpr`)**：
  构造 `q_idx = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)` 与 `k_idx = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)`，对于 `q_idx[:, None] < k_idx[None, :]` 的位置加上 `-1e6`。
- [ ] **验收 Triton 版前反向单测**：
  ```bash
  uv run pytest tests/test_attention.py -k "triton"
  ```

---

## 第三阶段：分布式数据并行 DDP (`cs336_systems/ddp.py`)

### 1. 学习准备
- [ ] **看课**：观看 **Lecture 7**，精读 [../lectures/lecture_07.py](../lectures/lecture_07.py) 中的 `collective_operations()` 与 `data_parallelism()`。
- [ ] **读 PDF**：[cs336_assignment2_systems.pdf](./cs336_assignment2_systems.pdf) **Section 3 (Distributed Data Parallel Training)**。

### 2. 核心实现细节与避坑点 (`get_ddp` & `ddp_on_after_backward`)
仔细看 [tests/test_ddp.py](./tests/test_ddp.py) 的测试逻辑：
1. **初始化权重广播（`__init__`）**：
   - 各卡（rank）刚创建模型时随机种子不同，参数不一致。
   - 在 `DDP(module)` 的 `__init__` 中，遍历 `self.module.parameters()`（以及 buffers），调用 `dist.broadcast(param.data, src=0)` 将 rank 0 的初始参数同步给所有 rank！
2. **注册反向传播异步通信 Hook（Communication-Computation Overlap）**：
   - 注意共享权重（`ToyModelWithTiedWeights`）！同一个 `nn.Parameter` 对象可能在模型中出现多次，遍历 `self.module.parameters()` 默认会自动去重，但为安全起见可用 `seen_params` 集合去重，且**只对 `param.requires_grad == True` 的参数注册 hook**。
   - 使用 `param.register_post_accumulate_grad_hook(hook_fn)`：
     当某个参数的梯度在反向传播中累加完毕时，立刻在 `hook_fn(p)` 中：
     1. 将 `p.grad.data` 除以 `world_size`（或者在 `all_reduce` 完成后除以 `world_size`）；
     2. 发起**异步**归约：`handle = dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM, async_op=True)`；
     3. 将 `handle` 存入列表 `self.handles.append(handle)`。
3. **`ddp_on_after_backward(ddp_model, optimizer)`**：
   - 遍历 `ddp_model.handles`，对每个 `handle` 调用 `handle.wait()` 等待所有异步通信完成，并清空 `handles.clear()`（如果之前未除以 `world_size`，在此处统一对每个 `p.grad` 除以 `world_size`）。
- [ ] **验收 DDP 单测**：
  ```bash
  uv run pytest tests/test_ddp.py
  ```

---

## 第四阶段：优化器状态分片 (ZeRO-1) 与 FSDP (`sharded_optimizer.py` & `fsdp.py`)

### 1. 学习准备
- [ ] **看课**：回顾 [../lectures/lecture_07.py](../lectures/lecture_07.py) 中的 ZeRO Stage 1 / Stage 3 (FSDP) 原理，以及 [../lectures/lecture_08.pdf](../lectures/lecture_08.pdf)。
- [ ] **读 PDF**：[cs336_assignment2_systems.pdf](./cs336_assignment2_systems.pdf) **Section 4**。

### 2. 核心实现细节

#### 任务 4.1：优化器状态分片 `ShardedOptimizer` (`get_sharded_optimizer`)
- 继承 `torch.optim.Optimizer`。
- 将模型的所有参数列表按 `world_size` 划分（例如第 `i` 个参数归属 `owner_rank = i % world_size`）。
- 当前 `rank` 只把自己负责的那部分参数传给底层的 `optimizer_cls(my_params, **kwargs)`，从而让每个 rank 只分配 $1 / \text{world\_size}$ 的 AdamW 状态（`m` 和 `v`）显存！
- 在 `step(closure=None)` 中：
  1. 先调用底层局部优化器的 `self.local_optimizer.step(closure)` 更新当前 rank 负责的参数；
  2. 遍历所有参数，由每个参数的 `owner_rank` 调用 `dist.broadcast(p.data, src=owner_rank)` 将更新后的最新权重同步给其他所有 rank！
- [ ] **验收 `ShardedOptimizer` 单测**：
  ```bash
  uv run pytest tests/test_sharded_optimizer.py
  ```

#### 任务 4.2：完全分片数据并行 `FSDP` (`get_fsdp`, `fsdp_on_after_backward`, `fsdp_gather_full_params`)
- 查看 [tests/test_fsdp.py](./tests/test_fsdp.py) 了解 FSDP 的测试要求：
  - 将可分片层（参数元素总数能被 `world_size` 整除的层）的权重在第 0 维（或展平后）切分为 `world_size` 份，每个 rank 平时只存储 `1 / world_size` 的分片参数。
  - 在模块 `forward` 前（通过 `register_forward_pre_hook`）调用 `dist.all_gather` 拼回完整权重（若指定了 `compute_dtype` 则转为 `compute_dtype` 计算），在 `forward` 后释放完整权重；
  - 在 `backward` 时再次 `all_gather` 完整权重计算梯度，算完后用 `dist.reduce_scatter` 将梯度平均并切分回各 rank 对应的分片梯度 `p.grad`。
- [ ] **验收 FSDP 单测**：
  ```bash
  uv run pytest tests/test_fsdp.py
  ```
- [ ] **Assignment 2 全量单测总验收**：
  ```bash
  uv run pytest
  ```
