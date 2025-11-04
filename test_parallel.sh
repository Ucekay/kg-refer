#!/bin/bash

echo "=== Testing EDC Parallel Processing ==="
echo "This test uses a small dataset to verify parallel processing functionality"
echo ""

# Test with parallel processing enabled
echo "1. Testing PARALLEL processing mode..."
uv run --env-file .env --package edc python packages/edc/main.py \
    --input_text_file_path ./packages/edc/datasets/test_parallel.txt \
    --oie_llm gpt-4.1-nano \
    --sd_llm gpt-4.1-nano \
    --sc_llm gpt-4.1-nano \
    --ee_llm gpt-4.1-nano \
    --sc_embedder intfloat/e5-mistral-7b-instruct \
    --enable_parallel_processing \
    --max_concurrent_requests 2 \
    --max_requests_per_second 10 \
    --request_timeout 30.0 \
    --output_dir ./packages/edc/output/test_parallel

echo ""
echo "2. Testing SEQUENTIAL processing mode for comparison..."
uv run --env-file .env --package edc python packages/edc/main.py \
    --input_text_file_path ./packages/edc/datasets/test_parallel.txt \
    --oie_llm gpt-4.1-nano \
    --sd_llm gpt-4.1-nano \
    --sc_llm gpt-4.1-nano \
    --ee_llm gpt-4.1-nano \
    --sc_embedder intfloat/e5-mistral-7b-instruct \
    --output_dir ./packages/edc/output/test_sequential

echo ""
echo "=== Test completed! ==="
echo "Compare results in:"
echo "- Parallel: ./packages/edc/output/test_parallel"
echo "- Sequential: ./packages/edc/output/test_sequential"
