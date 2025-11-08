#!/bin/bash

# Yelp dataset processing with EDC framework (TEST VERSION)
# - Uses first 100 items for quick testing
# - refinement_iterations = 0 (no refinement)
# - No target schema (self-canonicalization)
# - Schema enrichment enabled
# - Async processing for parallel execution

uv run --env-file .env --package edc python packages/edc/main.py \
    --input-text-file-path ./data/yelp/item_profile_test_10.jsonl \
    --oie-llm gpt-4.1-nano \
    --oie-input-template-file-path ./packages/edc/prompt_templates/oie_input_template_yelp.txt \
    --oie-instructions-template-file-path ./packages/edc/prompt_templates/oie_instructions_template_yelp.txt \
    --oie-few-shot-example-file-path ./packages/edc/few_shot_examples/yelp/oie_few_shot_examples.txt \
    --target-schema-path ./packages/edc/schemas/yelp_schema.csv \
    --sd-llm gpt-4.1-nano \
    --sd-input-template-file-path ./packages/edc/prompt_templates/sd_input_template_yelp.txt \
    --sd-instructions-template-file-path ./packages/edc/prompt_templates/sd_instructions_template_yelp.txt \
    --sd-few-shot-example-file-path ./packages/edc/few_shot_examples/yelp/sd_few_shot_examples.txt \
    --sc-llm gpt-4.1-nano \
    --sc-input-template-file-path ./packages/edc/prompt_templates/sc_input_template_yelp.txt \
    --sc-instructions-template-file-path ./packages/edc/prompt_templates/sc_instructions_template_yelp.txt \
    --sc-embedder intfloat/e5-mistral-7b-instruct \
    --refinement-iterations 0 \
    --enrich-schema \
    --enable-parallel-requests \
    --output-dir ./packages/edc/output/yelp_test_with_schema_4.1 \
