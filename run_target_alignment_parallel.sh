# !/bin/bash

# 並列処理実行スクリプト (gpt-4.1-nano)

# 設定
DATASET="${1:-rebel}"
MAX_CONCURRENT="${MAX_CONCURRENT:-100}"
MAX_RPS="${MAX_REQUESTS_PER_SECOND:-600}"
SC_EMBEDDER="${3:-intfloat/e5-mistral-7b-instruct}"
SR_ADAPTER_PATH="${4:-}"

# 実行
uv run --env-file .env --package edc python packages/edc/main.py \
    --input_text_file_path packages/edc/datasets/${DATASET}.txt \
    --target_schema_path packages/edc/schemas/${DATASET}_schema.csv \
    --oie_llm gpt-4.1-nano \
    --sd_llm gpt-4.1-nano \
    --sc_llm gpt-4.1-nano \
    --ee_llm gpt-4.1-nano \
    --sc_embedder intfloat/e5-mistral-7b-instruct \
    --oie_few_shot_example_file_path packages/edc/few_shot_examples/${DATASET}/oie_few_shot_examples.txt \
    --oie_prompt_template_file_path packages/edc/prompt_templates/oie_template_responses_v2.txt \
    --sd_few_shot_example_file_path packages/edc/few_shot_examples/${DATASET}/sd_few_shot_examples.txt \
    --sd_prompt_template_file_path packages/edc/prompt_templates/sd_template_responses_v2.txt \
    --sc_prompt_template_file_path packages/edc/prompt_templates/sc_template_responses_v2.txt \
    --refined_oie_few_shot_example_file_path packages/edc/few_shot_examples/${DATASET}/oie_few_shot_refine_examples.txt \
    --refined_oie_prompt_template_file_path packages/edc/prompt_templates/oie_r_template_responses_v2.txt \
    --enable_parallel_processing \
    --max_concurrent_requests $MAX_CONCURRENT \
    --max_requests_per_second $MAX_RPS \
    --refinement_iterations 1 \
    --sr_embedder intfloat/e5-mistral-7b-instruct \
    --output_dir packages/edc/output/${DATASET}_parallel_gpt4nano
    --log_level debug
