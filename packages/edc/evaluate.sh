#!/bin/bash

# gpt-4.1-nano評価スクリプト
DATASET="${1:-rebel}"

EDC_OUTPUT="./output/${DATASET}_parallel_gpt4nano/iter1/canon_kg.txt"
REFERENCE="./evaluate/references/${DATASET}.txt"

echo "Evaluating gpt-4.1-nano results for ${DATASET}..."
uv run python evaluate/evaluation_script.py --edc_output ${EDC_OUTPUT} --reference ${REFERENCE}
