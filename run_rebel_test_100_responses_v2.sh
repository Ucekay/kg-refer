uv run --env-file .env --package edc python packages/edc/main.py \
    --input_text_file_path ./packages/edc/datasets/rebel_test_100.txt \
    --target_schema_path ./packages/edc/schemas/rebel_schema.csv \
    --oie_llm gpt-4.1-nano \
    --oie_prompt_template_file_path ./packages/edc/prompt_templates/oie_template_responses_v2.txt \
    --oie_few_shot_example_file_path ./packages/edc/few_shot_examples/rebel/oie_few_shot_examples.txt \
    --sd_llm gpt-4.1-nano \
    --sd_prompt_template_file_path ./packages/edc/prompt_templates/sd_template_responses_v2.txt \
    --sd_few_shot_example_file_path ./packages/edc/few_shot_examples/rebel/sd_few_shot_examples.txt \
    --sc_llm gpt-4.1-nano \
    --sc_prompt_template_file_path ./packages/edc/prompt_templates/sc_template_responses_v2.txt \
    --sc_embedder intfloat/e5-mistral-7b-instruct \
    --ee_llm gpt-4.1-nano \
    --ee_prompt_template_file_path ./packages/edc/prompt_templates/ee_template_responses_v2.txt \
    --ee_few_shot_example_file_path ./packages/edc/few_shot_examples/rebel/ee_few_shot_examples.txt \
    --em_prompt_template_file_path ./packages/edc/prompt_templates/em_template_responses_v2.txt \
    --refined_oie_prompt_template_file_path ./packages/edc/prompt_templates/oie_r_template_responses_v2.txt \
    --refined_oie_few_shot_example_file_path ./packages/edc/few_shot_examples/rebel/oie_few_shot_refine_examples.txt \
    --refinement_iterations 1 \
    --output_dir ./packages/edc/output/rebel_test_100_responses_v2_gpt4_1_nano

echo "Rebel test 100 dataset execution with Responses API v2 templates completed!"
