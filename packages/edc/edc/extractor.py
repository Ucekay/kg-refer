from typing import List, Optional

from tqdm.asyncio import tqdm
from transformers import PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from edc.utils import llm_utils
from edc.utils.async_utils import AsyncOpenAIProcessor


class LocalExtractor:
    """Extractor for local Hugging Face models."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast):
        self.model = model
        self.tokenizer = tokenizer

    def extract(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        instructions_template_str: str,
        input_template_str: str,
        entities_hint: Optional[str] = None,
        relations_hint: Optional[str] = None,
        item_id: Optional[int] = None,
    ) -> List[List[str]]:
        filled_instructions = instructions_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
            }
        )
        filled_input = input_template_str.format_map(
            {
                "input_text": input_text_str,
                "entities_hint": entities_hint,
                "relations_hint": relations_hint,
                "item_id": item_id if item_id is not None else "",
            }
        )

        messages = [
            {"role": "system", "content": filled_instructions},
            {"role": "user", "content": filled_input},
        ]

        completion = llm_utils.generate_completion_transformers(
            messages, self.model, self.tokenizer, answer_prepend="Triplets: "
        )
        extracted_triplets_list = llm_utils.parse_raw_triplets(completion)
        return extracted_triplets_list


class OpenAIExtractor:
    """Extractor for OpenAI models."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def extract(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        instructions_template_str: str,
        input_template_str: str,
        entities_hint: Optional[str] = None,
        relations_hint: Optional[str] = None,
        item_id: Optional[int] = None,
    ) -> List[List[str]]:
        filled_instructions = instructions_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
            }
        )
        filled_input = input_template_str.format_map(
            {
                "input_text": input_text_str,
                "entities_hint": entities_hint,
                "relations_hint": relations_hint,
                "item_id": item_id if item_id is not None else "",
            }
        )

        completion = llm_utils.openai_chat_completion(
            self.model_name, filled_instructions, filled_input
        )
        extracted_triplets_list = llm_utils.parse_raw_triplets(completion)
        return extracted_triplets_list


class OpenAIAsyncExtractor:
    def __init__(self, model_name: str, max_concurrent=200, max_req_per_sec=80) -> None:
        self.model_name = model_name
        self.max_concurrent = max_concurrent
        self.max_req_per_sec = max_req_per_sec

    async def extract_with_schema_async(
        self,
        input_text_list: List[str],
        few_shot_examples_str: str,
        instructions_template_str: str,
        input_template_str: str,
        entities_hint_list: Optional[List[str]] = None,
        relations_hint_list: Optional[List[str]] = None,
        input_ids_list: Optional[List] = None,
    ):
        filled_instructions = instructions_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
            }
        )
        filled_input_list = [
            input_template_str.format_map(
                {
                    "input_text": input_text,
                    "entities_hint": entities_hint_list[i]
                    if entities_hint_list
                    else None,
                    "relations_hint": relations_hint_list[i]
                    if relations_hint_list
                    else None,
                    "item_id": input_ids_list[i] if input_ids_list else "",
                }
            )
            for i, input_text in enumerate(input_text_list)
        ]

        async_openai_processor = AsyncOpenAIProcessor(
            max_concurrent=self.max_concurrent,
            max_req_per_sec=self.max_req_per_sec,
        )

        tasks = [
            async_openai_processor.get_parsed_triplets_async(
                self.model_name,
                instructions=filled_instructions,
                input=filled_input,
            )
            for filled_input in filled_input_list
        ]
        results = await tqdm.gather(*tasks, desc="Extracting triplets asynchronously")
        extracted_triplets_lists = [
            llm_utils.parse_schema_triplets(result) for result in results
        ]
        return extracted_triplets_lists

    async def extract_async(
        self,
        input_text_list: List[str],
        few_shot_examples_str: str,
        instructions_template_str: str,
        input_template_str: str,
        entities_hint_list: Optional[List[str]] = None,
        relations_hint_list: Optional[List[str]] = None,
        input_ids_list: Optional[List] = None,
    ) -> List[List[List[str]]]:
        filled_instructions = instructions_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
            }
        )
        filled_input_list = [
            input_template_str.format_map(
                {
                    "input_text": input_text,
                    "entities_hint": entities_hint_list[i]
                    if entities_hint_list
                    else None,
                    "relations_hint": relations_hint_list[i]
                    if relations_hint_list
                    else None,
                    "item_id": input_ids_list[i] if input_ids_list else "",
                }
            )
            for i, input_text in enumerate(input_text_list)
        ]

        async_openai_processor = AsyncOpenAIProcessor(
            max_concurrent=self.max_concurrent,
            max_req_per_sec=self.max_req_per_sec,
        )

        tasks = [
            async_openai_processor.openai_responses_async(
                self.model_name,
                instructions=filled_instructions,
                input=filled_input,
            )
            for filled_input in filled_input_list
        ]
        results = await tqdm.gather(*tasks, desc="Extracting triplets asynchronously")
        extracted_triplets_lists = [
            llm_utils.parse_raw_triplets(result) for result in results
        ]
        return extracted_triplets_lists
