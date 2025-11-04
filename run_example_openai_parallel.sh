#!/bin/bash

# EDC Framework Example with Parallel Processing (OpenAI)
# This script runs the example dataset with parallel processing enabled

echo "Starting EDC Framework with Parallel Processing..."
echo "Models: gpt-4.1-nano for all OpenAI models"
echo "Embedder: intfloat/e5-mistral-7b-instruct"
echo "Parallel Processing: ENABLED"
echo ""

uv run --env-file .env --package edc python packages/edc/main.py \
    --oie_llm gpt-4.1-nano \
    --sd_llm gpt-4.1-nano \
    --sc_llm gpt-4.1-nano \
    --ee_llm gpt-4.1-nano \
    --sc_embedder intfloat/e5-mistral-7b-instruct \
    --enable_parallel_processing \
    --max_concurrent_requests 5 \
    --max_requests_per_second 100 \
    --request_timeout 30.0 \
    --output_dir ./packages/edc/output/example_target_alignment_gpt4_1_nano_parallel

echo ""
echo "Parallel processing example completed!"
echo "Results saved to: ./packages/edc/output/example_target_alignment_gpt4_1_nano_parallel"
echo ""
echo "To compare with sequential processing, run:"
echo "./run_example_openai.sh"
