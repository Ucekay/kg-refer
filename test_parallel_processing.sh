#!/bin/bash

# 並列処理機能テスト用スクリプト
# 小規模データセットで並列処理の動作確認


echo "=============================================="
echo "EDC Parallel Processing Test"
echo "=============================================="

# テスト用の小規模データセットを使用
TEST_DATASET="test_parallel"
TEST_SIZE=5  # 5つのテキストでテスト

# 並列処理設定（gpt-4.1-nano最適化）
MAX_CONCURRENT=10
MAX_REQUESTS_PER_SECOND=200
REQUEST_TIMEOUT=30

echo "Test Configuration:"
echo "  Dataset: ${TEST_DATASET} (${TEST_SIZE} texts)"
echo "  Model: gpt-4.1-nano"
echo "  Max Concurrent: ${MAX_CONCURRENT}"
echo "  Max RPM: ${MAX_REQUESTS_PER_SECOND}"
echo "  Timeout: ${REQUEST_TIMEOUT}s"
echo "=============================================="

# 小規模テストデータがなければ作成
if [ ! -f "packages/edc/datasets/${TEST_DATASET}.txt" ]; then
    echo "Creating test dataset..."
    cat > "packages/edc/datasets/${TEST_DATASET}.txt" << EOF
John Doe is a student at National University of Singapore majoring in computer science.
Alice Smith works as a software engineer at Google's headquarters in Mountain View, California.
The University of Tokyo was founded in 1877 and is located in Bunkyo, Tokyo, Japan.
Dr. Robert Johnson is a professor of machine learning at MIT's Computer Science and Artificial Intelligence Laboratory.
The Eiffel Tower was built in 1889 and stands 324 meters tall in Paris, France.
EOF
    echo "Test dataset created with ${TEST_SIZE} texts."
fi

# 小規模テスト用のスキーマファイルも作成
if [ ! -f "packages/edc/schemas/${TEST_DATASET}_schema.csv" ]; then
    echo "Creating test schema..."
    mkdir -p packages/edc/schemas
    cat > "packages/edc/schemas/${TEST_DATASET}_schema.csv" << EOF
student,A person who is studying at a school or college
works at,The place where someone is employed
located in,To be situated in a particular place
founded in,The year when an organization was established
professor,A teacher of the highest rank in a university or college
built in,The year when a structure was constructed
EOF
    echo "Test schema created."
fi

echo ""
echo "Starting parallel processing test..."
echo "This test will process ${TEST_SIZE} texts in parallel mode."
echo ""

# 並列処理テストの実行
uv run --env-file .env --package edc python packages/edc/main.py \
    --input_text_file_path packages/edc/datasets/${TEST_DATASET}.txt \
    --oie_llm gpt-4.1-nano \
    --sd_llm gpt-4.1-nano \
    --sc_llm gpt-4.1-nano \
    --ee_llm gpt-4.1-nano \
    --sc_embedder intfloat/e5-mistral-7b-instruct \
    --enable_parallel_processing \
    --max_concurrent_requests ${MAX_CONCURRENT} \
    --max_requests_per_second ${MAX_REQUESTS_PER_SECOND} \
    --refinement_iterations 1 \
    --sr_embedder intfloat/e5-mistral-7b-instruct \
    --output_dir ./packages/edc/output/test_parallel_small
