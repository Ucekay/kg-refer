#!/bin/bash

# シンプルな並列処理実行スクリプト
# gpt-4.1-nanoで並列処理を有効にして実行

# .envファイル読み込み
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# APIキーチェック
if [ -z "$OPENAI_KEY" ]; then
    echo "Error: Set OPENAI_KEY in .env file"
    exit 1
fi

# 設定
DATASET="${1:-test_parallel}"
MAX_CONCURRENT="${MAX_CONCURRENT:-30}"
MAX_RPS="${MAX_RPS:-600}"

echo "Running: $DATASET (gpt-4.1-nano, parallel: $MAX_CONCURRENT concurrent)"

# 実行
uv run --env-file .env --package edc python packages/edc/main.py \
    --input_text_file_path packages/edc/datasets/${DATASET}.txt \
    --oie_llm gpt-4.1-nano \
    --sd_llm gpt-4.1-nano \
    --sc_llm gpt-4.1-nano \
    --ee_llm gpt-4.1-nano \
    --sc_embedder intfloat/e5-mistral-7b-instruct \
    --enable_parallel_processing \
    --max_concurrent_requests $MAX_CONCURRENT \
    --max_requests_per_second $MAX_RPS \
    --refinement_iterations 1 \
    --output_dir ./output/${DATASET}_parallel_gpt4nano

echo "Done: ./output/${DATASET}_parallel_gpt4nano"
