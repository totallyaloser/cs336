# Assignment 3: Scaling Laws — 详细通关实战指南

> **本目录目标**：掌握大模型预训练中的核心科学方法——**缩放定律（Scaling Laws）**。通过拟合 IsoFLOPs 实验曲线与小算力实验，在有限算力预算下科学预测大算力预算（如 $10^{23}$ / $10^{24}$ FLOPs 或给定 GPU 时间预算）下的**最优模型参数规模 $N_{\text{opt}}$、最优训练数据量 $D_{\text{opt}}$、最优架构宽高比与最优学习率**。

---

## 📁 目录核心文件导览

```text
assignment3-scaling/
├── cs336_assignment3_scaling.pdf          # 作业说明文档
├── data/
│   └── isoflops_curves.json               # [阶段 1 核心数据] 预跑好的各算力梯度下 (parameters, compute_budget, final_loss) 数据点
├── examples/
│   └── client_example.ipynb               # 实验配置提交、查询与结果可视化示例 Notebook
├── cs336_scaling/
│   ├── training/                          # 本地可直接运行的训练引擎 (BasicTransformerConfig, AdamWConfig, run.py)
│   └── scheduler/                         # 实验调度器与 API 服务
└── tests/
    ├── test_api.py
    └── test_scheduler.py
```

---

## 第一阶段：看课与理论准备（Chinchilla Scaling Laws 推导）

### 1. 学习准备
- [ ] **看课**：
  - 观看 **Lecture 9 (Scaling Laws I)** 与 **Lecture 11 (Scaling Laws II & $\mu$P)**，阅读课件 [../lectures/lecture_09.pdf](../lectures/lecture_09.pdf) 和 [../lectures/lecture_11.pdf](../lectures/lecture_11.pdf)。
  - 观看 **Lecture 10 (Inference)**，阅读可执行讲义 [../lectures/lecture_10.py](../lectures/lecture_10.py)（了解推理阶段的算术强度与 KV Cache 开销如何反过来影响训练时的 Chinchilla 最优比选择——即为什么现代小模型往往选择 Over-training）。
- [ ] **读 PDF**：完整阅读 [cs336_assignment3_scaling.pdf](./cs336_assignment3_scaling.pdf)。

### 2. 必须搞懂的 3 个核心公式
1. **Transformer 训练算力与参数/Token 的近似关系**：
   $$C \approx 6 N D \implies D = \frac{C}{6 N}$$
   其中 $C$ 为总训练算力（FLOPs），$N$ 为非词嵌入（或总）参数量，$D$ 为训练 Token 总数。
2. **IsoFLOPs 抛物线拟合法（Chinchilla Approach 2）**：
   对于每一个固定的算力预算 $C_i$（如 $6 \times 10^{18}, 10^{19}, 3 \times 10^{19}, \dots$），在对数坐标 $\log_{10} N$ 下，验证集损失 $L$ 呈开口向上的二次曲线（U 型曲线）：
   $$L(\log_{10} N) \approx a (\log_{10} N)^2 + b (\log_{10} N) + c$$
   曲线的最低点对应的横坐标即为该算力预算 $C_i$ 下的最优参数量：
   $$\log_{10} N_{\text{opt}}(C_i) = -\frac{b}{2a}$$
3. **幂律外推公式（Power-Law Extrapolation）**：
   拿到多个算力梯度 $(C_i, N_{\text{opt}}(C_i))$ 和 $(C_i, D_{\text{opt}}(C_i))$ 后，拟合幂律关系：
   $$N_{\text{opt}}(C) = k_N \cdot C^{\alpha}, \quad D_{\text{opt}}(C) = k_D \cdot C^{\beta}$$
   （在对数空间即线性回归：$\log N_{\text{opt}} = \alpha \log C + \log k_N$，且理论上 $\alpha + \beta \approx 1$）。

---

## 第二阶段：基于 `data/isoflops_curves.json` 拟合 Scaling Law

### 动手实践步骤（建议新建 `fit_isoflops.py` 或在 Notebook 中完成）

- [ ] **Step 1：加载并按 `compute_budget` 分组数据**
  - 读取 [data/isoflops_curves.json](./data/isoflops_curves.json)。
  - 按 `compute_budget` ($C \in \{6\times 10^{18}, 10^{19}, 3\times 10^{19}, 6\times 10^{19}, 10^{20}, 3\times 10^{20}, 6\times 10^{20}, 10^{21}, 3\times 10^{21}\}$) 分组。
  - 计算每个数据点对应的训练 Token 数：$D = \frac{C}{6 N}$。

- [ ] **Step 2：提取每个 $C_i$ 的最优 $(N_{\text{opt}}, D_{\text{opt}}, L_{\min})$**
  - 方法 A（离散最小值法）：直接取每个 `compute_budget` 组内 `final_loss` 最小的数据点（或对 $\log N$ 做二次多项式拟合 `np.polyfit(np.log10(N), loss, deg=2)` 求顶点）。
  - 绘制全部 9 条 IsoFLOPs 曲线（横轴 $\log_{10} N$，纵轴 `final_loss`），并在每条曲线上标出最低点 $N_{\text{opt}}(C_i)$。

- [ ] **Step 3：拟合幂律指数 $\alpha, \beta$ 并外推预测**
  - 使用 `scipy.optimize.curve_fit` 或 `np.polyfit(np.log10(C_list), np.log10(N_opt_list), deg=1)`：
    - 拟合 $N_{\text{opt}}(C) = k_N C^\alpha$，画出数据点与拟合直线，预测当 $C = 10^{23}$ 和 $C = 10^{24}$ FLOPs 时的最优模型大小 $N^*$。
    - 拟合 $D_{\text{opt}}(C) = k_D C^\beta$，预测当 $C = 10^{23}$ 和 $C = 10^{24}$ FLOPs 时的最优数据集大小 $D^*$。
    - 拟合最优损失与算力的幂律 $L_{\text{opt}}(C) = E + A_C C^{-\gamma}$，预测目标算力下的最终 Loss。

---

## 第三阶段：实验设计与本地 RTX 5090 D 实测拟合

由于你是本地自学（无需连接斯坦福校内 `hyperturing.stanford.edu` 服务器），你可以直接调用本仓库内置的训练引擎 [cs336_scaling/training/run.py](./cs336_scaling/training/run.py) 在你的 **RTX 5090 D** 上跑小规模 Scaling 实验！

- [ ] **Step 1：跑通仓库自带测试**
  ```bash
  uv run pytest
  ```
- [ ] **Step 2：阅读 [examples/client_example.ipynb](./examples/client_example.ipynb) 与 [cs336_scaling/training/training_config.py](./cs336_scaling/training/training_config.py)**
  - 了解 `TrainingConfig`、`BasicTransformerConfig` 和 `AdamWConfig` 的超参数空间：
    - 架构参数：`hidden_size` ($d_{\text{model}}$)、`num_hidden_layers` ($L$)、`num_attention_heads`、`head_dim`、`intermediate_size` ($d_{\text{ff}}$)。
    - 优化器参数：`peak_value` (LR)、`train_batch_size`、`warmup_frac`、`total_train_tokens`。
- [ ] **Step 3：掌握模型架构黄金法则（Aspect Ratio & Learning Rate Scaling）**
  1. **深宽比（Aspect Ratio $d_{\text{model}} / n_{\text{layers}}$）**：通常保持在 $32 \sim 64$ 之间最稳定；`intermediate_size` 约取 $\frac{8}{3} d_{\text{model}}$ 并对齐到 64 或 128 的倍数；`head_dim` 固定为 64。
  2. **学习率随模型宽度缩放**：当模型参数量变大（$d_{\text{model}}$ 变宽）时，最优峰值学习率 $\text{lr}_{\text{peak}}$ 随 $d_{\text{model}}$ 增大而减小（近似 $\propto d_{\text{model}}^{-0.5} \sim d_{\text{model}}^{-1}$）。
  3. **Batch Size 选择**：算力越充裕，临界 Batch Size（Critical Batch Size）越大，可适当调大 `train_batch_size` 提升 GPU 吞吐率（Tokens/sec）。
