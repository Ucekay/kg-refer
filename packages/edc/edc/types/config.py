"""Configuration types for EDC framework."""

import logging
from dataclasses import dataclass
from typing import Literal, Optional

from cyclopts import Parameter


@Parameter(name="*")
@dataclass
class EDCConfig:
    """Configuration for EDC (Extractive Data Completion) framework."""

    # Open Information Extraction
    oie_llm: str = "mistralai/Mistral-7B-Instruct-v0.2"
    oie_instructions_template_file_path: str = (
        "./packages/edc/prompt_templates/oie_instructions_template.txt"
    )
    oie_input_template_file_path: str = (
        "./packages/edc/prompt_templates/oie_input_template.txt"
    )
    oie_few_shot_example_file_path: str = (
        "./packages/edc/few_shot_examples/example/oie_few_shot_examples.txt"
    )

    # Schema Definition
    sd_llm: str = "mistralai/Mistral-7B-Instruct-v0.2"
    sd_instructions_template_file_path: str = (
        "./packages/edc/prompt_templates/sd_instructions_template.txt"
    )
    sd_input_template_file_path: str = (
        "./packages/edc/prompt_templates/sd_input_template.txt"
    )
    sd_few_shot_example_file_path: str = (
        "./packages/edc/few_shot_examples/example/sd_few_shot_examples.txt"
    )

    # Schema Canonicalization
    sc_llm: str = "mistralai/Mistral-7B-Instruct-v0.2"
    sc_embedder: str = "intfloat/e5-mistral-7b-instruct"
    sc_instructions_template_file_path: str = (
        "./packages/edc/prompt_templates/sc_instructions_template.txt"
    )
    sc_input_template_file_path: str = (
        "./packages/edc/prompt_templates/sc_input_template.txt"
    )

    # Schema Retriever
    sr_adapter_path: Optional[str] = None
    sr_embedder: str = "intfloat/e5-mistral-7b-instruct"

    # Refined Open Information Extraction
    refined_oie_instructions_template_file_path: str = (
        "./packages/edc/prompt_templates/oie_r_instructions_template.txt"
    )
    refined_oie_input_template_file_path: str = (
        "./packages/edc/prompt_templates/oie_r_input_template.txt"
    )
    refined_oie_few_shot_example_file_path: str = (
        "./packages/edc/few_shot_examples/example/r_oie_few_shot_examples.txt"
    )

    # Entity Extraction
    ee_llm: str = "mistralai/Mistral-7B-Instruct-v0.2"
    ee_instructions_template_file_path: str = (
        "./packages/edc/prompt_templates/ee_instructions_template.txt"
    )
    ee_input_template_file_path: str = (
        "./packages/edc/prompt_templates/ee_input_template.txt"
    )
    ee_few_shot_example_file_path: str = (
        "./packages/edc/few_shot_examples/example/ee_few_shot_examples.txt"
    )

    # Entity Matching
    em_instructions_template_file_path: str = (
        "./packages/edc/prompt_templates/em_instructions_template.txt"
    )
    em_input_template_file_path: str = (
        "./packages/edc/prompt_templates/em_input_template.txt"
    )

    # Input/Output
    input_text_file_path: str = "./packages/edc/datasets/example.txt"
    target_schema_path: str = "./packages/edc/schemas/example_schema.csv"
    output_dir: str = "./packages/edc/output/tmp"

    # Processing options
    refinement_iterations: int = 0
    enrich_schema: bool = False
    enable_parallel_requests: bool = False

    # Logging
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
