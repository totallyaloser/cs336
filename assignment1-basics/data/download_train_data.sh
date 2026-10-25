#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=== Downloading TinyStories Training Data (~2.2 GB) ==="
wget -c https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt

echo "=== Downloading OpenWebText Sample Training Data (~4.5 GB compressed) ==="
wget -c https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
echo "=== Uncompressing owt_train.txt.gz ==="
gunzip -f owt_train.txt.gz

echo "=== Done! All training data ready in $DIR ==="
