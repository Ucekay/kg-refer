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

        input, instructions = llm_utils.convert_to_responses_format(filled_prompt)

        completion = llm_utils.openai_chat_completion(
            self.model_name, instructions, input
        )
        extracted_triplets_list = llm_utils.parse_raw_triplets(completion)
        return extracted_triplets_list
