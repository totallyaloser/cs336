"""
BPE Tokenizer 训练 —— 多进程分块预分词 + 最大堆（Lazy Deletion）优化参考实现
================================================================================

设计思路与核心优化点（对照你原有的实现）：
1. 【阶段 1：多进程 + 流式分块预分词】
   - 问题：直接 `f.read()` 读入 2.1GB (TinyStories) 或 12GB (OpenWebText) 会导致内存暴涨甚至 OOM，
     且单进程正则分词在大语料上耗时数十分钟。
   - 思路：利用 `find_chunk_boundaries` 按 `<|endoftext|>` 找到安全的字节切分边界，将大文件划分为
     多个独立小块（例如每块 ~32MB），交由 `multiprocessing.Pool` 多进程并行读取、切分 special_tokens
     并运行 `re.finditer` 统计局部词频 `Counter`，最后在主进程汇总。
   - 边界修复：当 `special_tokens` 为空列表时，跳过 `re.split`，避免 `re.split("", text)` 将文本按单字符切碎。

2. 【阶段 2：构建词表数组 + 倒排索引 + 初始最大堆】
   - 保持你原有的 `words`、`counts`、`pair_counter`、`pair_to_word_ids` 结构。
   - 新增：将所有初始 `(count, pair)` 封装为 `_PairItem` 并通过 `heapq.heapify` 一次性建堆（O(P) 复杂度）。

3. 【阶段 4：基于最大堆 + 惰性删除 (Lazy Deletion) + 批量变更集 (changed_pairs) 的 Merge 循环】
   - 瓶颈 1 解决：用自定义 `__lt__` 的 `_PairItem` 配合 `heapq` 替代每轮 `max(pair_counter, ...)`。
     每次取堆顶时，检查堆顶记录的 `count` 是否与 `pair_counter[pair]` 当前最新值一致：
     若不一致（说明该 pair 后来发生过增减，此条堆记录已过期），直接丢弃并继续弹出，直到拿到有效最高频 pair。
   - 瓶颈 2 解决：在更新 `affected_word_ids` 时，不在内层循环中途 `del pair_counter[p]` 或频繁入堆，
     而是用 `changed_pairs` 集合记录本轮所有频次发生变动的 pair。等当前 `best_pair` 涉及的所有词都合并完后，
     统一遍历一次 `changed_pairs`：清零的删除，大于 0 的推入堆中（每个变动 pair 每轮最多入堆 1 次）。
"""

from __future__ import annotations

import heapq
import os
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from typing import BinaryIO
import regex as re

# GPT-2 预分词正则表达式
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RX = re.compile(PAT)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    来自 cs336_basics/pretokenization_example.py 的分块边界查找函数。
    （注：原文件末尾带有未包裹在 __main__ 中的 `with open(..., "rb")` 示例代码，直接 import 会报错，故在此处独立定义）
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
    if chunk_size == 0:
        return [0, file_size]

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)
        while True:
            mini_chunk = file.read(mini_chunk_size)
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    return sorted(set(chunk_boundaries))


class _PairItem:
    """
    用于 heapq 的包装类，实现最大堆排序语义：
    1. 优先按频次 `count` 降序（count 越大越靠前 -> `self.count > other.count`）
    2. 频次相同时，按 `pair` 字典序降序（字典序越大越靠前 -> `self.pair > other.pair`）
    使用 __slots__ 消除实例 __dict__ 开销，大幅降低内存占用并加快比较速度。
    """

    __slots__ = ("count", "pair")

    def __init__(self, count: int, pair: tuple[bytes, bytes]) -> None:
        self.count = count
        self.pair = pair

    def __lt__(self, other: _PairItem) -> bool:
        if self.count != other.count:
            return self.count > other.count
        return self.pair > other.pair


def _pretokenize_chunk_worker(
    args: tuple[str | os.PathLike, int, int, list[str]],
) -> Counter[str]:
    """
    子进程 Worker：只读取文件中 [start, end) 字节区间，完成局部预分词并返回局部 Counter。
    """
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        raw_bytes = f.read(end - start)

    text = raw_bytes.decode("utf-8", errors="ignore")

    # 注意处理 special_tokens 为空的情况，避免 re.split("", text) 把字符全部拆散
    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        special_pat = "|".join(re.escape(tok) for tok in sorted_specials)
        sub_chunks = re.split(special_pat, text)
    else:
        sub_chunks = [text]

    local_counter: Counter[str] = Counter()
    for chunk in sub_chunks:
        if chunk:
            local_counter.update(m.group(0) for m in PAT_RX.finditer(chunk))
    return local_counter


def pretokenize_corpus(
    input_path: str | os.PathLike,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> Counter[str]:
    """
    阶段 1：支持单进程（小文件快速路径）与多进程分块（大文件低内存路径）的预分词统计。
    """
    file_size = os.path.getsize(input_path)

    if num_processes is None:
        num_processes = os.cpu_count() or 4

    # 如果文件很小（例如 < 2MB 的单元测试语料）或没有 special_tokens 切分点，直接单进程处理，省去进程池启动开销
    if file_size < 2 * 1024 * 1024 or num_processes <= 1 or not special_tokens:
        return _pretokenize_chunk_worker((input_path, 0, file_size, special_tokens))

    # 大文件：按第一个 special_token (如 <|endoftext|>) 寻找分块边界
    # 将总块数设为使每块大约 32MB（同时不少于进程数的 2 倍），兼顾负载均衡与低内存峰值
    split_token_bytes = special_tokens[0].encode("utf-8")
    desired_chunks = max(num_processes * 2, file_size // (32 * 1024 * 1024))
    desired_chunks = min(desired_chunks, 256)

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_chunks, split_token_bytes)

    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
        if end > start
    ]

    raw_word_counter: Counter[str] = Counter()
    with Pool(processes=min(num_processes, len(tasks))) as pool:
        for chunk_counter in pool.imap_unordered(_pretokenize_chunk_worker, tasks):
            raw_word_counter.update(chunk_counter)

    return raw_word_counter


def train_bpe_optimized(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    多进程预分词 + 最大堆优化版 BPE 训练主函数。
    """
    # ==========================================
    # 阶段 1：多进程预分词与字符串级频次统计
    # ==========================================
    raw_word_counter = pretokenize_corpus(
        input_path=input_path,
        special_tokens=special_tokens,
        num_processes=num_processes,
    )

    # ==========================================
    # 阶段 2：构建核心索引结构（词表列表 + 倒排索引 + 初始 Pair 计数）
    # ==========================================
    words: list[list[bytes]] = []
    counts: list[int] = []

    pair_counter: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_word_ids: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    # 预先构建 0..255 的单字节对象表，避免内层循环重复调用 bytes([b])
    byte_table = [bytes([i]) for i in range(256)]

    for word_id, (raw_word, count) in enumerate(raw_word_counter.items()):
        raw_word_bytes = [byte_table[b] for b in raw_word.encode("utf-8")]
        words.append(raw_word_bytes)
        counts.append(count)

        for idx in range(len(raw_word_bytes) - 1):
            pair = (raw_word_bytes[idx], raw_word_bytes[idx + 1])
            pair_counter[pair] += count
            pair_to_word_ids[pair].add(word_id)

    # ==========================================
    # 阶段 3：初始化最大堆（O(P) 线性时间建堆）
    # ==========================================
    heap: list[_PairItem] = [
        _PairItem(cnt, pair) for pair, cnt in pair_counter.items() if cnt > 0
    ]
    heapq.heapify(heap)

    # ==========================================
    # 阶段 4：BPE 核心 Merge 循环（堆加速 + 批量增量更新）
    # ==========================================
    merges: list[tuple[bytes, bytes]] = []
    merge_tokens: list[bytes] = []
    iter_nums = max(vocab_size - 256 - len(special_tokens), 0)

    for _ in range(iter_nums):
        # --- 步骤 4.1：从最大堆弹出当前最高频 best_pair（惰性跳过已过期的旧记录） ---
        best_pair: tuple[bytes, bytes] | None = None
        while heap:
            top = heapq.heappop(heap)
            # 校验堆顶快照频次是否等于当前真实频次
            if pair_counter.get(top.pair, 0) == top.count and top.count > 0:
                best_pair = top.pair
                break

        if best_pair is None:
            break

        part1, part2 = best_pair
        best_token = part1 + part2
        merges.append(best_pair)
        merge_tokens.append(best_token)

        # --- 步骤 4.2：通过倒排索引，只更新包含 best_pair 的词 ---
        affected_word_ids = pair_to_word_ids.pop(best_pair, set())
        pair_counter.pop(best_pair, None)

        # 记录本轮频次发生过变化的 pair，避免对同一个 pair 在每个词里重复入堆
        changed_pairs: set[tuple[bytes, bytes]] = set()

        for word_id in affected_word_ids:
            affected_word = words[word_id]
            count = counts[word_id]

            # 1) 扣减旧词中所有相邻对的频次（暂不 del，避免随后立即重建 key）
            for i in range(len(affected_word) - 1):
                old_pair = (affected_word[i], affected_word[i + 1])
                pair_counter[old_pair] -= count
                pair_to_word_ids[old_pair].discard(word_id)
                changed_pairs.add(old_pair)

            # 2) 扫描合并生成新词
            new_word: list[bytes] = []
            i = 0
            n = len(affected_word)
            while i < n:
                if (
                    i < n - 1
                    and affected_word[i] == part1
                    and affected_word[i + 1] == part2
                ):
                    new_word.append(best_token)
                    i += 2
                else:
                    new_word.append(affected_word[i])
                    i += 1
            words[word_id] = new_word

            # 3) 增加新词中所有相邻对的频次
            for i in range(len(new_word) - 1):
                new_pair = (new_word[i], new_word[i + 1])
                pair_counter[new_pair] += count
                pair_to_word_ids[new_pair].add(word_id)
                changed_pairs.add(new_pair)

        # --- 步骤 4.3：统一清理归零项，并将真正发生变化的 pair 最新频次推入堆 ---
        changed_pairs.discard(best_pair)
        for pair in changed_pairs:
            curr_cnt = pair_counter.get(pair, 0)
            if curr_cnt <= 0:
                pair_counter.pop(pair, None)
                pair_to_word_ids.pop(pair, None)
            else:
                heapq.heappush(heap, _PairItem(curr_cnt, pair))

    # 构建最终词表
    rt_lst = (
        [spt.encode("utf-8") for spt in special_tokens]
        + byte_table
        + merge_tokens
    )
    vocab: dict[int, bytes] = {i: b for i, b in enumerate(rt_lst)}
    return vocab, merges


if __name__ == "__main__":
    # 对比基准测试：在 5MB 样本与 22MB 验证集上对比原版与优化版耗时及结果一致性
    from pathlib import Path
    from tests.adapters import run_train_bpe

    base_dir = Path(__file__).resolve().parents[2]
    benchmarks = [
        ("5MB Sample (vocab=1000)", base_dir / "tests" / "fixtures" / "tinystories_sample_5M.txt", 1000),
        ("22MB TinyStories Valid (vocab=10000)", base_dir / "data" / "TinyStoriesV2-GPT4-valid.txt", 10000),
    ]

    for name, path, vsize in benchmarks:
        if not path.exists():
            continue
        t0 = time.perf_counter()
        v1, m1 = run_train_bpe(path, vocab_size=vsize, special_tokens=["<|endoftext|>"])
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        v2, m2 = train_bpe_optimized(path, vocab_size=vsize, special_tokens=["<|endoftext|>"])
        t3 = time.perf_counter()

        assert v1 == v2 and m1 == m2, f"{name} 结果不一致！"
        print(f"[{name}]")
        print(f"  原版耗时  : {t1 - t0:.3f}s")
        print(f"  优化版耗时: {t3 - t2:.3f}s (提速 {(t1 - t0) / (t3 - t2):.2f}x)")
        print("  词表与 Merges 校验：100% 完全一致！\n")
