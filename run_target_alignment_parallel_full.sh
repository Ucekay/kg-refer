#!/bin/bash

# 並列処理実行スクリプト (gpt-4.1-nano + 完全版)

# .envファイル読み込み
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# 設定可能な変数
OIE_LLM="${1:-gpt-4.1-nano}"
DATASET="${2:-rebel}"
SC_EMBEDDER="${3:-intfloat/e5-mistral-7b-instruct}"
SR_ADAPTER_PATH="${4:-}"
REFINEMENT_ITERATIONS="${5:-1}"

# gpt-4.1-nano用の設定
SD_LLM="gpt-4.1-nano"
SC_LLM="gpt-4.1-nano"
EE_LLM="gpt-4.1-nano"

# 並列処理設定（安全なデフォルト値）
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
MAX_RPS="${MAX_REQUESTS_PER_SECOND:-200}"

# refinementオプションの構築
REFINEMENT_OPTIONS=""
if [ ! -z "$SR_ADAPTER_PATH" ]; then
    REFINEMENT_OPTIONS="--sr_adapter_path $SR_ADAPTER_PATH"
else
    REFINEMENT_OPTIONS="--sr_embedder intfloat/e5-mistral-7b-instruct"
fi

# 実行
uv run --env-file .env --package edc python packages/edc/main.py \
    --input_text_file_path packages/edc/datasets/${DATASET}.txt \
    --oie_llm ${OIE_LLM} \
    --oie_few_shot_example_file_path packages/edc/few_shot_examples/${DATASET}/oie_few_shot_examples.txt \
    --oie_prompt_template_file_path packages/edc/prompt_templates/oie_template_responses_v2.txt \
    --sd_llm ${SD_LLM} \
    --sd_few_shot_example_file_path packages/edc/few_shot_examples/${DATASET}/sd_few_shot_examples.txt \
    --sd_prompt_template_file_path packages/edc/prompt_templates/sd_template_responses_v2.txt \
    --sc_llm ${SC_LLM} \
    --sc_embedder ${SC_EMBEDDER} \
    --sc_prompt_template_file_path packages/edc/prompt_templates/sc_template_responses_v2.txt \
    --target_schema_path packages/edc/schemas/${DATASET}_schema.csv \
    --oie_refine_few_shot_example_file_path packages/edc/few_shot_examples/${DATASET}/oie_few_shot_refine_examples.txt \
    --oie_refine_prompt_template_file_path packages/edc/prompt_templates/oie_r_template_responses_v2.txt \
    --ee_llm ${EE_LLM} \
    --ee_few_shot_example_file_path packages/edc/few_shot_examples/${DATASET}/ee_few_shot_examples.txt \
    --ee_prompt_template_file_path packages/edc/prompt_templates/ee_template_responses_v2.txt \
    --em_prompt_template_file_path packages/edc/prompt_templates/em_template_responses_v2.txt \
    ${REFINEMENT_OPTIONS} \
    --refinement_iterations ${REFINEMENT_ITERATIONS} \
    --output_dir ./output/${DATASET}_target_alignment_parallel_gpt4nano_${REFINEMENT_ITERATIONS} \
    --enable_parallel_processing \
    --max_concurrent_requests $MAX_CONCURRENT \
    --max_requests_per_second $MAX_RPS

echo "Target Alignment with parallel processing completed!"
echo "Dataset: ${DATASET}"
echo "Using model: gpt-4.1-nano for all LLM components"
echo "Parallel: ${MAX_CONCURRENT} concurrent, ${MAX_RPS} RPM"
echo "Refinement iterations: ${REFINEMENT_ITERATIONS}"
echo "Output: ./output/${DATASET}_target_alignment_parallel_gpt4nano_${REFINEMENT_ITERATIONS}"
