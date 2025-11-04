import asyncio
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
    ) -> Any | list[Any]:
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
        messages = [{"role": "user", "content": filled_prompt}]

        completion = llm_utils.openai_chat_completion(self.model_name, None, messages)
        extracted_entities = llm_utils.parse_raw_entities(completion)
        return extracted_entities

    async def extract_entities_async(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        prompt_template_str: str,
    ):
        """Async version of extract_entities for parallel processing."""
        filled_prompt = prompt_template_str.format_map(
            {"few_shot_examples": few_shot_examples_str, "input_text": input_text_str}
        )
        messages = [{"role": "user", "content": filled_prompt}]

        completion = await llm_utils.openai_chat_completion_async(
            self.model_name, None, messages
        )
        extracted_entities = llm_utils.parse_raw_entities(completion)
        return extracted_entities

    async def extract_entities_batch_async(
        self,
        input_text_list: List[str],
        few_shot_examples_str: str,
        prompt_template_str: str,
        max_concurrent: int = 5,
        max_requests_per_second: int = 200,
    ):
        """Batch async version for processing multiple texts in parallel."""
        # Prepare prompts for each input
        prompts = []
        for input_text_str in input_text_list:
            filled_prompt = prompt_template_str.format_map(
                {
                    "few_shot_examples": few_shot_examples_str,
                    "input_text": input_text_str,
                }
            )
            prompts.append(filled_prompt)

        # Process prompts in parallel
        responses = await llm_utils.process_prompts_with_openai_async(
            model=self.model_name,
            prompts=prompts,
            max_concurrent=max_concurrent,
            max_requests_per_second=max_requests_per_second,
        )

        # Parse responses
        results = []
        for response in responses:
            if response:  # Non-empty response
                extracted_entities = llm_utils.parse_raw_entities(response)
            else:  # Failed request
                extracted_entities = []
            results.append(extracted_entities)

        return results

    def merge_entities(
        self,
        input_text: str,
        entity_list_1: List[str],
        entity_list_2: List[str],
        prompt_template_str: str,
    ) -> Any | list[Any]:
        filled_prompt = prompt_template_str.format_map(
            {
                "input_text": input_text,
                "entity_list_1": entity_list_1,
                "entity_list_2": entity_list_2,
            }
        )
        messages = [{"role": "user", "content": filled_prompt}]

        completion = llm_utils.openai_chat_completion(self.model_name, None, messages)
        extracted_entities = llm_utils.parse_raw_entities(completion)
        return extracted_entities

    async def merge_entities_async(
        self,
        input_text: str,
        entity_list_1: List[str],
        entity_list_2: List[str],
        prompt_template_str: str,
    ):
        """Async version of merge_entities for parallel processing."""
        filled_prompt = prompt_template_str.format_map(
            {
                "input_text": input_text,
                "entity_list_1": entity_list_1,
                "entity_list_2": entity_list_2,
            }
        )
        messages = [{"role": "user", "content": filled_prompt}]

        completion = await llm_utils.openai_chat_completion_async(
            self.model_name, None, messages
        )
        extracted_entities = llm_utils.parse_raw_entities(completion)
        return extracted_entities

    async def merge_entities_batch_async(
        self,
        input_text_list: List[str],
        entity_list_1_list: List[List[str]],
        entity_list_2_list: List[List[str]],
        prompt_template_str: str,
        max_concurrent: int = 5,
        max_requests_per_second: int = 200,
    ):
        """Batch async version for merging multiple entity lists in parallel."""
        if not (
            len(input_text_list) == len(entity_list_1_list) == len(entity_list_2_list)
        ):
            raise ValueError("All input lists must have the same length")

        # Prepare prompts for each merge operation
        prompts = []
        for i, input_text in enumerate(input_text_list):
            filled_prompt = prompt_template_str.format_map(
                {
                    "input_text": input_text,
                    "entity_list_1": entity_list_1_list[i],
                    "entity_list_2": entity_list_2_list[i],
                }
            )
            prompts.append(filled_prompt)

        # Process prompts in parallel
        responses = await llm_utils.process_prompts_with_openai_async(
            model=self.model_name,
            prompts=prompts,
            max_concurrent=max_concurrent,
            max_requests_per_second=max_requests_per_second,
        )

        # Parse responses
        results = []
        for response in responses:
            if response:  # Non-empty response
                extracted_entities = llm_utils.parse_raw_entities(response)
            else:  # Failed request
                extracted_entities = []
            results.append(extracted_entities)

        return results
