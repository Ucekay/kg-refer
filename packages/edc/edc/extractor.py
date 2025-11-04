import asyncio
from abc import ABC, abstractmethod
from typing import List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from edc.utils import llm_utils


class BaseExtractor(ABC):
    """Base class for all extractors."""

    @abstractmethod
    def extract(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        prompt_template_str: str,
        entities_hint: Optional[str] = None,
        relations_hint: Optional[str] = None,
    ) -> List[List[str]]:
        """Extract triplets from input text using few-shot examples."""
        pass


class LocalExtractor(BaseExtractor):
    """Extractor for local Hugging Face models."""

    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast):
        self.model = model
        self.tokenizer = tokenizer

    def extract(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        prompt_template_str: str,
        entities_hint: Optional[str] = None,
        relations_hint: Optional[str] = None,
    ) -> List[List[str]]:
        filled_prompt = prompt_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
                "input_text": input_text_str,
                "entities_hint": entities_hint,
                "relations_hint": relations_hint,
            }
        )

        messages = [{"role": "user", "content": filled_prompt}]

        completion = llm_utils.generate_completion_transformers(
            messages, self.model, self.tokenizer, answer_prepend="Triplets: "
        )
        extracted_triplets_list = llm_utils.parse_raw_triplets(completion)
        return extracted_triplets_list


class OpenAIExtractor(BaseExtractor):
    """Extractor for OpenAI models."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def extract(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        prompt_template_str: str,
        entities_hint: Optional[str] = None,
        relations_hint: Optional[str] = None,
    ) -> List[List[str]]:
        filled_prompt = prompt_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
                "input_text": input_text_str,
                "entities_hint": entities_hint,
                "relations_hint": relations_hint,
            }
        )

        messages = [{"role": "user", "content": filled_prompt}]

        completion = llm_utils.openai_chat_completion(self.model_name, None, messages)
        extracted_triplets_list = llm_utils.parse_raw_triplets(completion)
        return extracted_triplets_list

    async def extract_async(
        self,
        input_text_str: str,
        few_shot_examples_str: str,
        prompt_template_str: str,
        entities_hint: Optional[str] = None,
        relations_hint: Optional[str] = None,
    ) -> List[List[str]]:
        """Async version of extract for parallel processing."""
        filled_prompt = prompt_template_str.format_map(
            {
                "few_shot_examples": few_shot_examples_str,
                "input_text": input_text_str,
                "entities_hint": entities_hint,
                "relations_hint": relations_hint,
            }
        )

        messages = [{"role": "user", "content": filled_prompt}]

        completion = await llm_utils.openai_chat_completion_async(
            self.model_name, None, messages
        )
        extracted_triplets_list = llm_utils.parse_raw_triplets(completion)
        return extracted_triplets_list

    async def extract_batch_async(
        self,
        input_text_list: List[str],
        few_shot_examples_str: str,
        prompt_template_str: str,
        entities_hints: Optional[List[str]] = None,
        relations_hints: Optional[List[str]] = None,
        max_concurrent: int = 5,
        max_requests_per_second: int = 200,
    ) -> List[List[List[str]]]:
        """Batch async version for processing multiple texts in parallel."""
        # Prepare prompts for each input
        prompts = []
        for i, input_text_str in enumerate(input_text_list):
            entities_hint = entities_hints[i] if entities_hints else None
            relations_hint = relations_hints[i] if relations_hints else None

            filled_prompt = prompt_template_str.format_map(
                {
                    "few_shot_examples": few_shot_examples_str,
                    "input_text": input_text_str,
                    "entities_hint": entities_hint,
                    "relations_hint": relations_hint,
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
                extracted_triplets_list = llm_utils.parse_raw_triplets(response)
            else:  # Failed request
                extracted_triplets_list = []
            results.append(extracted_triplets_list)

        return results
