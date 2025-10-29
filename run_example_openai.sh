

uv run --env-file .env --package edc python packages/edc/main.py \
    --oie_llm gpt-4.1-nano \
    --sd_llm gpt-4.1-nano \
    --sc_llm gpt-4.1-nano \
    --ee_llm gpt-4.1-nano \
    --sc_embedder intfloat/e5-mistral-7b-instruct \
    --output_dir ./packages/edc/output/example_target_alignment_gpt4_1_nano

echo "Example dataset execution completed!"
