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
        extracted_triplets_list: List[str],
        few_shot_examples_str: str,
        prompt_template_str: str,
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
        extracted_triplets_list: List[str],
        few_shot_examples_str: str,
        prompt_template_str: str,
    ) -> Dict[str, str]:
        # Given a piece of text and a list of triplets extracted from it, define each of the relation present
        relations_present = set()
        for t in extracted_triplets_list:
            relations_present.add(t[1])

        filled_prompt = prompt_template_str.format_map(
            {
                "text": input_text_str,
                "few_shot_examples": few_shot_examples_str,
                "relations": relations_present,
                "triples": extracted_triplets_list,
            }
        )
        messages = [{"role": "user", "content": filled_prompt}]

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
        extracted_triplets_list: List[str],
        few_shot_examples_str: str,
        prompt_template_str: str,
    ) -> Dict[str, str]:
        # Given a piece of text and a list of triplets extracted from it, define each of the relation present
        relations_present: set[str] = set()
        for t in extracted_triplets_list:
            relations_present.add(t[1])

        filled_prompt = prompt_template_str.format_map(
            {
                "text": input_text_str,
                "few_shot_examples": few_shot_examples_str,
                "relations": relations_present,
                "triples": extracted_triplets_list,
            }
        )
        messages = [{"role": "user", "content": filled_prompt}]

        completion = llm_utils.openai_chat_completion(self.model_name, None, messages)

        relation_definition_dict = llm_utils.parse_relation_definition(completion)

        missing_relations = [
            rel for rel in relations_present if rel not in relation_definition_dict
        ]
        if len(missing_relations) != 0:
            logger.debug(
                f"Relations {missing_relations} are missing from the relation definition!"
            )
        return relation_definition_dict


# Keep the old class for backward compatibility (deprecated)
class SchemaDefiner:
    """Deprecated: Use LocalSchemaDefiner or OpenAISchemaDefiner instead."""

    def __init__(
        self,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        openai_model: Optional[str] = None,
    ) -> None:
        assert openai_model is not None or (model is not None and tokenizer is not None)

        if openai_model is not None:
            self._definer = OpenAISchemaDefiner(openai_model)
        else:
            self._definer = LocalSchemaDefiner(model, tokenizer)

    def define_schema(
        self,
        input_text_str: str,
        extracted_triplets_list: List[str],
        few_shot_examples_str: str,
        prompt_template_str: str,
    ) -> Dict[str, str]:
        return self._definer.define_schema(
            input_text_str,
            extracted_triplets_list,
            few_shot_examples_str,
            prompt_template_str,
        )
