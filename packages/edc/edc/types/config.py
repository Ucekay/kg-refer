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
    oie_prompt_template_file_path: str = (
        "./packages/edc/prompt_templates/oie_template.txt"
    )
    oie_few_shot_example_file_path: str = (
        "./packages/edc/few_shot_examples/example/oie_few_shot_examples.txt"
    )

    # Schema Definition
    sd_llm: str = "mistralai/Mistral-7B-Instruct-v0.2"
    sd_prompt_template_file_path: str = (
        "./packages/edc/prompt_templates/sd_template.txt"
    )
    sd_few_shot_example_file_path: str = (
        "./packages/edc/few_shot_examples/example/sd_few_shot_examples.txt"
    )

    # Schema Canonicalization
    sc_llm: str = "mistralai/Mistral-7B-Instruct-v0.2"
    sc_embedder: str = "intfloat/e5-mistral-7b-instruct"
    sc_prompt_template_file_path: str = (
        "./packages/edc/prompt_templates/sc_template.txt"
    )

    # Schema Retriever
    sr_adapter_path: Optional[str] = None
    sr_embedder: str = "intfloat/e5-mistral-7b-instruct"

    # Refined Open Information Extraction
    refined_oie_prompt_template_file_path: str = (
        "./packages/edc/prompt_templates/r_oie_template.txt"
    )
    refined_oie_few_shot_example_file_path: str = (
        "./packages/edc/few_shot_examples/example/r_oie_few_shot_examples.txt"
    )

    # Entity Extraction
    ee_llm: str = "mistralai/Mistral-7B-Instruct-v0.2"
    ee_prompt_template_file_path: str = (
        "./packages/edc/prompt_templates/ee_template.txt"
    )
    ee_few_shot_example_file_path: str = (
        "./packages/edc/few_shot_examples/example/ee_few_shot_examples.txt"
    )

    # Entity Matching
    em_prompt_template_file_path: str = (
        "./packages/edc/prompt_templates/em_template.txt"
    )

    # Input/Output
    input_text_file_path: str = "./packages/edc/datasets/example.txt"
    target_schema_path: str = "./packages/edc/schema/example_schema.csv"
    output_dir: str = "./packages/edc/output/tmp"

    # Processing options
    refinement_iterations: int = 0
    enrich_schema: bool = False

    # Logging
    log_level: Literal["debug", "info", "warning", "error", "critical"] = "info"
