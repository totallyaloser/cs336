import json
from tests.adapters import run_train_bpe
from pathlib import Path
import time

"""
Problem (train_bpe_tinystories):  BPE Training on TinyStories (2 points)
(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary
size of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the
vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How
much time and memory did training take? What is the longest token in the vocabulary? Does
it make sense?
Resource requirements: ≤ 30 minutes (no GPUs), ≤ 30 GB RAM
9
Hint  You should be able to get under 2 minutes for BPE training using multiprocessing
during pre-tokenization and the following two facts:
(a) The <|endoftext|> token delimits documents in the data files.
(b) The <|endoftext|> token is handled as a special case before the BPE merges are applied.
Deliverable: A one-to-two sentence response.
(b) Profile your code. What part of the tokenizer training process takes the most time?
Deliverable: A one-to-two sentence response.
"""
def train_bpe_tinystories():
    special_tokens = ["<|endoftext|>"]
    vocab_size = 10000
    base_dir = Path(__file__).resolve().parents[2]
    start_time = time.time()
    vocab, merges = run_train_bpe(base_dir / "data" / "TinyStoriesV2-GPT4-train.txt", vocab_size, special_tokens)
    serializable_vocab:dict[str, str] = {str(idx):token.hex() for idx,token in  vocab.items()}
    serializable_merges:list[tuple[str,str]] = [(pair[0].hex(), pair[1].hex()) for pair in merges]

    # 写入文件中
    with open("repo/vocab.json", "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)
    with open("repo/merges.json", "w", encoding="utf-8") as f:
        json.dump(serializable_merges, f, ensure_ascii=False, indent=2)
    end_time = time.time()
    print(f"cost time = f{end_time - start_time}")
if __name__ == '__main__':
    train_bpe_tinystories()