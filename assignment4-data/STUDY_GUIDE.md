# Assignment 4: Data — 详细通关实战指南

> **本目录目标**：从零搭建工业级预训练数据清洗与去重流水线（Common Crawl WARC/WET 处理 $\to$ HTML 提取 $\to$ 语言识别 $\to$ PII 隐私脱敏 $\to$ 有害内容与 Gopher 启发式过滤 $\to$ 质量分类器 $\to$ 精确行级去重与 MinHash-LSH 近似文档去重），并用你清洗出的语料在 **RTX 5090 D** 上训练 GPT-2 模型对比验证集 Perplexity。

---

## 📁 推荐你在 `cs336_data/` 下建立的工程结构

```text
assignment4-data/
├── cs336_data/
│   ├── __init__.py
│   ├── extract.py              # [模块 1] HTML 编码检测与正文提取 (resiliparse)
│   ├── langid.py               # [模块 2] fastText 语言识别
│   ├── pii.py                  # [模块 3] 邮箱 / 电话 / IPv4 正则脱敏
│   ├── toxicity.py             # [模块 4] NSFW 与 Toxic Speech 分类器
│   ├── quality.py              # [模块 5] Gopher 启发式过滤规则 + fastText 质量分类器
│   ├── deduplication.py        # [模块 6] 精确行级去重 + MinHash-LSH 模糊文档去重
│   └── pipeline.py             # [模块 7] 串联全流程过滤 Common Crawl WET 文件
└── tests/
    └── adapters.py             # 连接上述模块与 pytest 单元测试
```

---

## 前置准备：看课与下载分类器模型

- [ ] **看课**：
  - 观看 **Lecture 12 (Evaluation)**，阅读 [../lectures/lecture_12.py](../lectures/lecture_12.py)。
  - 观看 **Lecture 13 (Data I)** 与 **Lecture 14 (Data II)**，精读 [../lectures/lecture_13.py](../lectures/lecture_13.py) 和 [../lectures/lecture_14.py](../lectures/lecture_14.py)（特别是 `lecture_14.py` 里关于 fastText、MinHash 与 LSH 分桶的代码示例）。
- [ ] **读 PDF**：[cs336_assignment4_data.pdf](./cs336_assignment4_data.pdf) **Section 1 ~ Section 4**。
- [ ] **检查 / 下载预训练分类器模型**（见 [scripts/download_data.py](./scripts/download_data.py)）：
  - `lid.176.bin`（fastText 语种识别模型）
  - Dolma Jigsaw NSFW & Toxic Speech fastText 模型

---

## 模块 1：HTML 文本提取 (`cs336_data/extract.py`)

- [ ] **读 PDF**：**Section 2.1 (Extracting text from HTML)**。
- [ ] **实现细节 (`run_extract_text_from_html_bytes(html_bytes: bytes) -> str | None`)**：
  1. 先尝试 `html_bytes.decode("utf-8")`；如果抛出 `UnicodeDecodeError`，使用 `resiliparse.parse.encoding.detect_encoding(html_bytes)` 检测编码并解码（`errors="replace"`）。
  2. 调用 `resiliparse.extract.html2text.extract_plain_text(html_str, main_content=False)` 提取纯文本并返回。
- [ ] **单测验收**：
  ```bash
  uv run pytest tests/test_extract.py
  ```

---

## 模块 2：语种识别 (`cs336_data/langid.py`)

- [ ] **读 PDF**：**Section 2.2 (Language identification)**。
- [ ] **实现细节 (`run_identify_language(text: str) -> tuple[Any, float]`)**：
  1. 使用模块级全局变量懒加载 `fasttext.load_model(...)`（避免每次调用重复从磁盘读模型！）。
  2. **避坑点**：`fasttext.predict` 不接受包含换行符 `\n` 的字符串！调用前必须先把 `text.replace("\n", " ")` 替换成单行。
  3. `labels, probs = model.predict(clean_text, k=1)`，将返回的标签前缀 `"__label__"` 去掉（例如 `"__label__en"` $\to$ `"en"`），返回 `(label, float(probs[0]))`。
- [ ] **单测验收**：
  ```bash
  uv run pytest tests/test_langid.py
  ```

---

## 模块 3：个人隐私信息 PII 脱敏 (`cs336_data/pii.py`)

- [ ] **读 PDF**：**Section 2.3 (Masking PII)**，对照 [tests/test_pii.py](./tests/test_pii.py) 中的测试样例。
- [ ] **实现细节**（使用 `re.subn(pattern, replacement, text)`，它能直接同时返回 `(替换后的字符串, 替换次数)`！）：
  1. **`run_mask_emails(text: str) -> tuple[str, int]`**：
     - 匹配标准邮箱格式，替换为 `"|||EMAIL_ADDRESS|||"`。
  2. **`run_mask_phone_numbers(text: str) -> tuple[str, int]`**：
     - 兼容 `2831823829`、`(283)-182-3829`、`(283) 182 3829`、`283-182-3829` 等 10 位美式电话格式，替换为 `"|||PHONE_NUMBER|||"`。
  3. **`run_mask_ips(text: str) -> tuple[str, int]`**：
     - 匹配 IPv4 地址（4 段 `0~255` 数字），替换为 `"|||IP_ADDRESS|||"`。
- [ ] **单测验收**：
  ```bash
  uv run pytest tests/test_pii.py
  ```

---

## 模块 4：有害内容过滤 (`cs336_data/toxicity.py`)

- [ ] **读 PDF**：**Section 2.4 (Harmful content)**。
- [ ] **实现细节 (`run_classify_nsfw` & `run_classify_toxic_speech`)**：
  - 同样使用 `fasttext.load_model` 懒加载对应的 NSFW 模型和 Toxic 模型。
  - 将输入文本中的 `\n` 替换为空格后调用 `.predict(clean_text)`，去掉 `"__label__"` 前缀，返回 `(label, float(prob))`（如 `("nsfw", 0.98)` 或 `("non-nsfw", 0.99)`）。
- [ ] **单测验收**：
  ```bash
  uv run pytest tests/test_toxicity.py
  ```

---

## 模块 5：Gopher 启发式规则与质量分类器 (`cs336_data/quality.py`)

- [ ] **读 PDF**：**Section 2.5 (Gopher quality rules) & Section 2.6 (Quality classifier)**。
- [ ] **实现 `run_gopher_quality_filter(text: str) -> bool`**（严格对照 [tests/test_quality.py](./tests/test_quality.py#L31-L94) 的 4 条规则）：
  使用 `nltk.tokenize.word_tokenize(text)` 分词后，若违反以下任一条件则返回 `False`，全部满足返回 `True`：
  1. **非纯符号词数量**：包含至少一个非标点/主要字符的词数必须在 $[50, 100000]$ 之间（`< 50` 或 `> 100000` 返回 `False`）；
  2. **平均词长**：所有词的平均字符长度必须在 $[3, 10]$ 之间（`< 3` 或 `> 10` 返回 `False`）；
  3. **省略号结尾行占比**：按 `\n` 切分各行，以 `"..."` 结尾的行数占比不能超过 $30\%$（`> 0.3` 返回 `False`）；
  4. **含字母词占比**：包含至少一个字母（`any(c.isalpha() for c in w)`）的词占总词数的比例必须 $\ge 80\%$（`< 0.8` 返回 `False`）。
- [ ] **实现 `run_classify_quality(text: str) -> tuple[Any, float]`**：
  - 训练或加载 fastText 质量分类器（正样本为 Wikipedia 参考文本 `__label__wiki`，负样本为 Common Crawl 原始文本 `__label__cc`），返回 `("wiki" 或 "cc", float(score))`。
- [ ] **单测验收**：
  ```bash
  uv run pytest tests/test_quality.py
  ```

---

## 模块 6：精确行级去重与 MinHash-LSH 模糊文档去重 (`cs336_data/deduplication.py`)

- [ ] **读 PDF**：**Section 3.1 (Exact line deduplication) & Section 3.2 (MinHash + LSH fuzzy document deduplication)**。

### 任务 6.1：`run_exact_line_deduplication(input_files, output_directory)`
- **两遍扫描算法**：
  1. **第一遍**：遍历所有 `input_files`，按行读取并统计每一行（或其哈希值）在整个语料库中出现的总次数 `line_counts[line] += 1`。
  2. **第二遍**：再次遍历每个输入文件 `in_path`，在 `output_directory / in_path.name` 写出新文件，**只保留 `line_counts[line] == 1` 的唯一行**！

### 任务 6.2：`run_minhash_deduplication(input_files, num_hashes, num_bands, ngrams, jaccard_threshold, output_directory)`
- **完整 4 步流水线**：
  1. **文本规范化与 n-gram 集合提取**：对每篇文档进行 NFC 规范化、转小写、去标点、空白折叠后，提取词级（或字符级，按 PDF 规范）`ngrams` 集合 $S_d$。
  2. **计算 MinHash 签名矩阵**：用 `num_hashes` 个带不同 `seed` 的哈希函数（如 `mmh3.hash(ngram, seed=i, signed=False)`），对每个文档求 $h_i(S_d) = \min_{g \in S_d} \text{hash}(g, i)$，得到长度为 `num_hashes` 的签名向量。
  3. **LSH 分带分桶 (Banding)**：将长度为 `num_hashes` 的签名均分为 `num_bands` 个 band（每个 band 含 `r = num_hashes // num_bands` 个值）。若两篇文档在**任意一个 band** 上的 `r` 元组完全相同，则将它们加入候选相似对 `(doc_i, doc_j)`。
  4. **真实 Jaccard 校验与连通分量聚类（Union-Find / 图遍历）**：
     - 对候选对计算真实 n-gram 集合的 Jaccard 相似度 $\frac{|S_i \cap S_j|}{|S_i \cup S_j|}$（或者按 PDF 要求直接使用候选对/签名估计）。若 $\ge \text{jaccard\_threshold}$，则在图中连边（并查集 `union`）。
     - 每个连通分量（簇）中**只保留 1 篇文档**，其余重复文档剔除，将保留的文档原样写入 `output_directory`。
- [ ] **单测验收**：
  ```bash
  uv run pytest tests/test_deduplication.py
  ```
- [ ] **Assignment 4 全量单测总验收**：
  ```bash
  uv run pytest
  ```
