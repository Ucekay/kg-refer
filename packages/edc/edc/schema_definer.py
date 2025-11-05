import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

import edc.utils.llm_utils as llm_utils

logger = logging.getLogger(__name__)


class BaseSchemaDefiner(ABC):
    """Base class for all schema definers."""

    @abstractmethod
    def define_schema(
        self,
        input_text_str: str,
        extracted_triplets_list: List[List[str]],
        few_shot_examples_str: str,
        instructions_template_str: str,
        input_template_str: str,
    ) -> Dict[str, str]:
        """Define schema for relations present in the extracted triplets."""
        pass


class LocalSchemaDefiner(BaseSchemaDefiner):
    """Schema definer for local Hugging Face models."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast):
        self.model = model
        self.tokenizer = tokenizer

    def define_schema(
        self,
        input_text_str: str,
        extracted_triplets_list: List[List[str]],
        few_shot_examples_str: str,
        instructions_template_str: str,
        input_template_str: str,
    ) -> Dict[str, str]:
        # Given a piece of text and a list of triplets extracted from it, define each of the relation present
        relations_present = set()
        for t in extracted_triplets_list:
            relations_present.add(t[1])

        filled_instructions = instructions_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
            }
        )
        filled_input = input_template_str.format_map(
            {
                "text": input_text_str,
                "relations": relations_present,
                "triples": extracted_triplets_list,
            }
        )
        messages = [
            {"role": "system", "content": filled_instructions},
            {"role": "user", "content": filled_input},
        ]

        completion = llm_utils.generate_completion_transformers(
            messages, self.model, self.tokenizer, answer_prepend="Answer: "
        )

        relation_definition_dict = llm_utils.parse_relation_definition(completion)

        missing_relations = [
            rel for rel in relations_present if rel not in relation_definition_dict
        ]
        if len(missing_relations) != 0:
            logger.debug(
                f"Relations {missing_relations} are missing from the relation definition!"
            )
        return relation_definition_dict


class OpenAISchemaDefiner(BaseSchemaDefiner):
    """Schema definer for OpenAI models."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def define_schema(
        self,
        input_text_str: str,
        extracted_triplets_list: List[List[str]],
        few_shot_examples_str: str,
        instructions_template_str: str,
        input_template_str: str,
    ) -> Dict[str, str]:
        # Given a piece of text and a list of triplets extracted from it, define each of the relation present
        relations_present: set[str] = set()
        for t in extracted_triplets_list:
            relations_present.add(t[1])

        filled_instructions = instructions_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
            }
        )
        filled_input = input_template_str.format_map(
            {
                "text": input_text_str,
                "relations": relations_present,
                "triples": extracted_triplets_list,
            }
        )
        completion = llm_utils.openai_chat_completion(
            self.model_name, filled_instructions, filled_input
        )

        relation_definition_dict = llm_utils.parse_relation_definition(completion)

        missing_relations = [
            rel for rel in relations_present if rel not in relation_definition_dict
        ]
        if len(missing_relations) != 0:
            logger.debug(
                f"Relations {missing_relations} are missing from the relation definition!"
            )
        return relation_definition_dict
