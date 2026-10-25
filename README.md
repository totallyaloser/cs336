# Stanford CS336: Language Modeling from Scratch

[Stanford CS336 Course Homepage](https://cs336.stanford.edu/)

Welcome to the CS336 workspace! This repository contains all official course materials, lecture notes/code, and assignments (1 to 5) configured and verified with GPU acceleration on your local machine.

---

## 🚀 Environment & Hardware Status

- **GPU**: NVIDIA GeForce RTX 5090 D (32 GB VRAM)
- **CUDA Runtime / Driver**: CUDA 13.0 / 13.4
- **PyTorch**: PyTorch 2.11 with CUDA acceleration enabled
- **Package & Environment Manager**: [`uv`](https://github.com/astral-sh/uv) (CPython 3.12 managed virtualenvs)

Each assignment has its own dedicated, isolated virtual environment managed seamlessly by `uv`.

---

## 📁 Repository Structure

```text
cs336/
├── assignment1-basics/       # Assignment 1: Tokenizer, Architecture, Training Loop
│   ├── cs336_basics/         # Core LM implementation modules
│   ├── tests/                # Unit tests & adapters.py
│   ├── data/                 # TinyStories & OpenWebText datasets (valid sets downloaded)
│   └── cs336_assignment1_basics.pdf
│
├── assignment2-systems/      # Assignment 2: Systems, Triton Kernels & Parallelism
│   ├── cs336_systems/        # FlashAttention-2 Triton kernels, profiling, DDP
│   ├── cs336-basics/         # Reference baseline implementation
│   └── cs336_assignment2_systems.pdf
│
├── assignment3-scaling/      # Assignment 3: Empirical Scaling Laws
│   ├── cs336_scaling/        # Scaling law fitting and loss prediction
│   └── cs336_assignment3_scaling.pdf
│
├── assignment4-data/         # Assignment 4: Data Processing Pipeline
│   ├── cs336_data/           # Common Crawl filtering, PII, deduplication (MinHash LSH)
│   └── cs336_assignment4_data.pdf
│
├── assignment5-alignment/    # Assignment 5: Alignment, SFT & RLHF
│   ├── cs336_alignment/      # SFT, DPO, GRPO (Reasoning RL), GSM8K eval
│   ├── cs336_spring2026_assignment5_alignment.pdf
│   └── cs336_spring2026_assignment5_supplement_safety_rlhf.pdf
│
└── lectures/                 # Course lectures, slides (PDFs), and runnable scripts
    ├── lecture_03.pdf ... lecture_16.pdf
    └── lecture_01.py ... lecture_17.py
```

---

## ⚡ Quick Start: Assignment 1 (Basics)

### 1. Navigate to Assignment 1
```bash
cd assignment1-basics
```

### 2. Run Unit Tests
To run all tests:
```bash
uv run pytest
```

To run a specific module test:
```bash
uv run pytest tests/test_model.py
uv run pytest tests/test_tokenizer.py
uv run pytest tests/test_optimizer.py
```

All assignments use adapter patterns in `tests/adapters.py`. As you implement features in `cs336_basics/`, link them in `tests/adapters.py` to make the tests pass.

### 3. Training Data
Validation datasets are already pre-downloaded in `assignment1-basics/data/`:
- `TinyStoriesV2-GPT4-valid.txt` (~22 MB)
- `owt_valid.txt` (~277 MB)

When you are ready to train full models and need the full training corpora (~6.7 GB):
```bash
cd assignment1-basics/data
./download_train_data.sh
```

---

## 💡 Recommended Active Workspace

You can open this project directly in your editor:
- Path: `/home/yu/.gemini/antigravity/scratch/cs336`
- Or alias: `/home/yu/projects/learning/cs336`
