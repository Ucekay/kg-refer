from abc import ABC, abstractmethod
from typing import Any, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from edc.utils import llm_utils


class BaseEntityExtractor(ABC):
    """Base class for all entity extractors."""

    @abstractmethod
    def extract_entities(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        prompt_template_str: str,
    ) -> Any | list[Any]:
        """Extract entities from input text using few-shot examples."""
        pass


class LocalEntityExtractor(BaseEntityExtractor):
    """Entity extractor for local Hugging Face models."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast):
        self.model = model
        self.tokenizer = tokenizer

    def extract_entities(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        prompt_template_str: str,
    ):
        filled_prompt = prompt_template_str.format_map(
            {"few_shot_examples": few_shot_examples_str, "input_text": input_text_str}
        )
        messages = [{"role": "user", "content": filled_prompt}]

        completion = llm_utils.generate_completion_transformers(
            messages,
            model=self.model,
            tokenizer=self.tokenizer,
            answer_prepend="Entities: ",
        )
        extracted_entities = llm_utils.parse_raw_entities(completion)
        return extracted_entities

    def merge_entities(
        self,
        input_text: str,
        entity_list_1: List[str],
        entity_list_2: List[str],
        prompt_template_str: str,
    ) -> List[str]:
        filled_prompt = prompt_template_str.format_map(
            {
                "input_text": input_text,
                "entity_list_1": entity_list_1,
                "entity_list_2": entity_list_2,
            }
        )
        messages = [{"role": "user", "content": filled_prompt}]

        completion = llm_utils.generate_completion_transformers(
            messages, self.model, self.tokenizer, answer_prepend="Answer: "
        )
        extracted_entities = llm_utils.parse_raw_entities(completion)
        return extracted_entities


class OpenAIEntityExtractor(BaseEntityExtractor):
    """Entity extractor for OpenAI models."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def extract_entities(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        prompt_template_str: str,
    ):
        filled_prompt = prompt_template_str.format_map(
            {"few_shot_examples": few_shot_examples_str, "input_text": input_text_str}
        )

        input, instructions = llm_utils.convert_to_responses_format(filled_prompt)

        completion = llm_utils.openai_chat_completion(
            self.model_name, instructions, input
        )
        extracted_entities = llm_utils.parse_raw_entities(completion)
        return extracted_entities

    def merge_entities(
        self,
        input_text: str,
        entity_list_1: List[str],
        entity_list_2: List[str],
        prompt_template_str: str,
    ) -> List[str]:
        filled_prompt = prompt_template_str.format_map(
            {
                "input_text": input_text,
                "entity_list_1": entity_list_1,
                "entity_list_2": entity_list_2,
            }
        )

        input, instructions = llm_utils.convert_to_responses_format(filled_prompt)

        completion = llm_utils.openai_chat_completion(
            self.model_name, instructions, input
        )
        extracted_entities = llm_utils.parse_raw_entities(completion)
        return extracted_entities
