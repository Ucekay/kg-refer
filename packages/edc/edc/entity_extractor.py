from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

from tqdm.asyncio import tqdm
from transformers import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from edc.utils import llm_utils
from edc.utils.async_utils import AsyncOpenAIProcessor


class BaseEntityExtractor(ABC):
    """Base class for all entity extractors."""

    @abstractmethod
    def extract_entities(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        instructions_template_str: str,
        input_template_str: str,
    ) -> Any | list[Any]:
        """Extract entities from input text using few-shot examples."""
        pass

    @abstractmethod
    def merge_entities(
        self,
        input_text: str,
        entity_list_1: List[str],
        entity_list_2: List[str],
        instructions_template_str: str,
        input_template_str: str,
    ) -> List[str]:
        """Merge two entity lists."""
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
        instructions_template_str: str,
        input_template_str: str,
    ):
        filled_instructions = instructions_template_str.format_map(
            {"few_shot_examples": few_shot_examples_str}
        )
        filled_input = input_template_str.format_map({"input_text": input_text_str})
        messages = [
            {"role": "system", "content": filled_instructions},
            {"role": "user", "content": filled_input},
        ]

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
        instructions_template_str: str,
        input_template_str: str,
    ) -> List[str]:
        filled_instructions = instructions_template_str.format_map({})
        filled_input = input_template_str.format_map(
            {
                "input_text": input_text,
                "entity_list_1": entity_list_1,
                "entity_list_2": entity_list_2,
            }
        )
        messages = [
            {"role": "system", "content": filled_instructions},
            {"role": "user", "content": filled_input},
        ]

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
        instructions_template_str: str,
        input_template_str: str,
    ):
        filled_instructions = instructions_template_str.format_map(
            {"few_shot_examples": few_shot_examples_str}
        )
        filled_input = input_template_str.format_map({"input_text": input_text_str})

        completion = llm_utils.openai_chat_completion(
            self.model_name, filled_instructions, filled_input
        )
        extracted_entities = llm_utils.parse_raw_entities(completion)
        return extracted_entities

    def merge_entities(
        self,
        input_text: str,
        entity_list_1: List[str],
        entity_list_2: List[str],
        instructions_template_str: str,
        input_template_str: str,
    ) -> List[str]:
        filled_instructions = instructions_template_str.format_map({})
        filled_input = input_template_str.format_map(
            {
                "input_text": input_text,
                "entity_list_1": entity_list_1,
                "entity_list_2": entity_list_2,
            }
        )

        completion = llm_utils.openai_chat_completion(
            self.model_name, filled_instructions, filled_input
        )
        extracted_entities = llm_utils.parse_raw_entities(completion)
        return extracted_entities


class OpenAIAsyncEntityExtractor:
    def __init__(self, model_name: str, max_concurrent=5, max_req_per_sec=200) -> None:
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.max_req_per_sec = max_req_per_sec

    async def extract_entity_hint_list_async(
        self,
        input_text_list: List[str],
        previous_entities_and_relations: List[Tuple[List[str], List[str]]],
        entity_extraction_few_shot_examples: str,
        entity_extraction_instructions_template: str,
        entity_extraction_input_template: str,
        entity_merging_instructions_template: str,
        entity_merging_input_template: str,
    ) -> List[str]:
        entity_extraction_instructions = (
            entity_extraction_instructions_template.format_map(
                {"few_shot_examples": entity_extraction_few_shot_examples}
            )
        )
        tasks = [
            self.extract_entity_hint_async(
                input_text,
                entity_extraction_instructions,
                entity_extraction_input_template,
                previous_entities,  # Tupleの最初の要素
                entity_merging_instructions_template,
                entity_merging_input_template,
            )
            for input_text, (previous_entities, previous_relations) in zip(
                input_text_list, previous_entities_and_relations
            )
        ]

        results = await tqdm.gather(
            *tasks, desc="Extracting entity hints", total=len(tasks)
        )
        return results

    async def extract_entity_hint_async(
        self,
        input_text: str,
        entity_extraction_instructions: str,
        entity_extrantion_input_template: str,
        previous_entity_list: List[str],
        entity_merging_instructions: str,
        entity_merging_input_template: str,
    ) -> List[str]:
        extracted_entities = await self.extract_entities_async(
            input_text,
            entity_extraction_instructions,
            entity_extrantion_input_template,
        )
        merged_entities = await self.merge_entities_async(
            input_text,
            previous_entity_list,
            extracted_entities,
            entity_merging_instructions,
            entity_merging_input_template,
        )

        return merged_entities

    async def extract_entities_async(
        self,
        input_text: str,
        instructions: str,
        input_template: str,
    ) -> List[str]:
        filled_input = input_template.format_map({"input_text": input_text})

        async_openai_processor = AsyncOpenAIProcessor(
            max_concurrent=self.max_concurrent // 2,
            max_req_per_sec=self.max_req_per_sec // 2,
        )

        response = await async_openai_processor.openai_responses_async(
            self.model_name,
            instructions,
            filled_input,
        )

        extracted_entities = llm_utils.parse_raw_entities(response)
        return extracted_entities

    async def merge_entities_async(
        self,
        input_text: str,
        entity_list_1: List[str],
        entity_list_2: List[str],
        instructions: str,
        input_template: str,
    ) -> List[str]:
        filled_input = input_template.format_map(
            {
                "input_text": input_text,
                "entity_list_1": entity_list_1,
                "entity_list_2": entity_list_2,
            }
        )

        async_openai_processor = AsyncOpenAIProcessor(
            max_concurrent=self.max_concurrent // 2,
            max_req_per_sec=self.max_req_per_sec // 2,
        )

        response = await async_openai_processor.openai_responses_async(
            self.model_name,
            instructions,
            filled_input,
        )

        extracted_entities = llm_utils.parse_raw_entities(response)

        return extracted_entities
