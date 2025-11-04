import ast
import asyncio
import gc
import logging
import os
import time
from typing import Any, Dict, List, Optional

import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from torch.types import Tensor
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

logger = logging.getLogger(__name__)


def free_model(
    model: Optional[PreTrainedModel | SentenceTransformer] = None,
    tokenizer: Optional[PreTrainedTokenizerFast] = None,
):
    try:
        if model is None:
            return
        model.cpu()
        del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Failed to free model resources: {e}")


def get_embedding_e5mistral(model: SentenceTransformer, tokenizer, sentence, task=None):
    model.eval()
    device = model.device

    if task is not None:
        sentence = get_detailed_instruct(task, sentence)

    sentence = [sentence]

    max_length = 4096

    batch_dict = tokenizer(
        sentence,
        max_length=max_length - 1,
        return_attention_mask=False,
        padding=False,
        truncation=True,
    )

    batch_dict["input_ids"] = [
        input_ids + [tokenizer.eos_token_id] for input_ids in batch_dict["input_ids"]
    ]

    batch_dict = tokenizer.pad(
        batch_dict, padding=True, return_attention_mask=True, return_tensors="pt"
    )

    batch_dict.to(device)

    embeddings = model(**batch_dict).detach().cpu()

    assert len(embeddings) == 1

    return embeddings[0]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


def get_embedding_sts(
    model: SentenceTransformer, text: str, prompt_name=None, prompt=None
) -> Tensor:
    embedding = model.encode(text, prompt_name=prompt_name, prompt=prompt)
    return embedding


def is_model_openai(model_name: str) -> bool:
    return "gpt" in model_name.lower()


def parse_raw_entities(raw_entities: str):
    parsed_entities = []
    try:
        left_bracket_index = raw_entities.index("[")
        right_bracket_index = raw_entities.rindex("]") + 1
        try:
            parsed_entities = ast.literal_eval(
                raw_entities[left_bracket_index : right_bracket_index + 1]
            )
        except Exception as e:
            logger.debug(f"Failed to parse entities from {raw_entities}: {e}")
    except ValueError:
        logger.debug(f"No brackets found in entities string: {raw_entities}")
    logger.debug(f"Entities {raw_entities} parsed as {parsed_entities}")
    return parsed_entities


def parse_raw_triplets(raw_triplets: str) -> List[List[str]]:
    unmatched_left_bracket_indices = []
    matched_bracket_pairs = []

    collected_triplets = []
    for c_index, c in enumerate(raw_triplets):
        if c == "[":
            unmatched_left_bracket_indices.append(c_index)
        elif c == "]":
            if len(unmatched_left_bracket_indices) == 0:
                continue
            matched_left_bracket_index = unmatched_left_bracket_indices.pop()
            matched_bracket_pairs.append((matched_left_bracket_index, c_index))

    for l, r in matched_bracket_pairs:
        bracket_str = raw_triplets[l : r + 1]
        try:
            parsed_triplet = ast.literal_eval(bracket_str)
            if len(parsed_triplet) == 3 and all(
                isinstance(item, str) for item in parsed_triplet
            ):
                if all([e != "" and e != "_" for e in parsed_triplet]):
                    collected_triplets.append(parsed_triplet)
            elif not all([type(x) == type(parsed_triplet[0]) for x in parsed_triplet]):
                for e_index, e in enumerate(parsed_triplet):
                    if isinstance(e, list):
                        parsed_triplet[e_index] = ",".join(e)
                collected_triplets.append(parsed_triplet)
        except Exception as e:
            pass
    logger.debug(f"Triplets {raw_triplets} parsed as {collected_triplets}")
    return collected_triplets


def parse_relation_definition(raw_definitions: str):
    descriptions = raw_definitions.split("\n")
    relation_definition_dict = {}

    for description in descriptions:
        if ":" not in description:
            continue
        index_of_colon = description.index(":")
        relation = description[:index_of_colon].strip()

        relation_description = description[index_of_colon + 1 :].strip()

        if relation == "Answer":
            continue

        relation_definition_dict[relation] = relation_description

    logger.debug(
        f"Relation definitions {raw_definitions} parsed as {relation_definition_dict}"
    )
    return relation_definition_dict


def generate_completion_transformers(
    input: list,
    model: "PreTrainedModel",
    tokenizer: PreTrainedTokenizerFast,
    max_new_tokens=256,
    answer_prepend="",
):
    device = model.device
    tokenizer.pad_token = tokenizer.eos_token

    messages = (
        str(
            tokenizer.apply_chat_template(
                input, add_generation_prompt=True, tokenize=False
            )
        )
        + answer_prepend
    )

    model_inputs = tokenizer(
        messages, return_tensors="pt", padding=True, add_special_tokens=False
    ).to(device)

    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    generation = model.generate(**model_inputs, generation_config=generation_config)

    sequences = generation["sequences"]
    generated_ids = sequences[:, model_inputs["input_ids"].shape[1] :]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()

    logging.debug(f"Prompt:\n {messages}\n Result: {generated_texts}")
    return generated_texts


def openai_chat_completion(
    model: str,
    system_prompt: Optional[str],
    history,
    temperature: float = 0.0,
    max_tokens: int = 512,
):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    input, instructions = _convert_to_responses_format(system_prompt, history)

    response = None

    while response is None:
        try:
            params = {
                "model": model,
                "input": input,
                "temperature": temperature,
                "max_output_tokens": max_tokens,
            }

            if instructions:
                params["instructions"] = instructions

            response = client.responses.create(**params)
        except Exception as e:
            logging.warning(f"OpenAI API request failed with error: {e}. Retrying...")
            time.sleep(1)

        # レスポンス処理
    result = response.output_text
    logging.debug(
        f"Model: {model}\nInput: {input}\nInstructions: {instructions}\nResult: {result}"
    )
    return result


def _convert_to_responses_format(system_prompt, history):
    """
    Chat Completions形式からResponses API形式への変換
    instructionsとinputを分離して最適化
    """
    if not history:
        raise ValueError("History cannot be empty")

    # 単一ユーザーメッセージの場合（このプロジェクトの主なケース）
    if len(history) == 1 and history[0].get("role") == "user":
        user_content = history[0]["content"]

        # 新しいプロンプト形式の場合、instructionsとinputに分離
        if "INPUT_DATA_START" in user_content and "INPUT_DATA_END" in user_content:
            return _split_prompt_content(user_content, system_prompt)
        else:
            # 従来形式の場合、system_promptをinstructionsとして使用
            instructions = system_prompt
            input_data = user_content
            return input_data, instructions

    # 複数メッセージの場合（フォールバック）
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)

    return messages, None


def _split_prompt_content(content, system_prompt=None):
    """
    INPUT_DATA_START/ENDで区切られたプロンプトをinstructionsとinputに分離
    """
    if "INPUT_DATA_START" in content and "INPUT_DATA_END" in content:
        parts = content.split("INPUT_DATA_START")
        instructions_part = parts[0].strip()
        input_part = parts[1].replace("INPUT_DATA_END", "").strip()

        # system_promptがあれば追加
        if system_prompt:
            instructions = f"{system_prompt}\n\n{instructions_part}"
        else:
            instructions = instructions_part

        return input_part, instructions
    else:
        # 分離できない場合は従来通り
        return content, system_prompt


async def openai_chat_completion_async(
    model: str,
    system_prompt: Optional[str],
    history: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Async version of openai_chat_completion for parallel processing."""
    from .parallel_llm_utils import get_global_processor

    processor = get_global_processor()

    # Convert to messages format for parallel processing
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)

    result = await processor.generate_with_limits(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )

    if result["success"]:
        return result["data"]
    else:
        logger.error(
            f"Async OpenAI request failed: {result.get('error', 'Unknown error')}"
        )
        raise Exception(
            f"OpenAI API request failed: {result.get('error', 'Unknown error')}"
        )


async def openai_chat_completion_batch_async(
    model: str,
    system_prompts: List[Optional[str]],
    histories: List[List[Dict[str, str]]],
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_concurrent: int = 5,
    max_requests_per_second: int = 200,
) -> List[str]:
    """Batch async version of openai_chat_completion for parallel processing."""
    from .parallel_llm_utils import get_global_processor

    processor = get_global_processor(
        max_concurrent=max_concurrent,
        max_requests_per_second=max_requests_per_second,
    )

    # Prepare messages for each request
    messages_list = []
    for system_prompt, history in zip(system_prompts, histories):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        messages_list.append(messages)

    result = await processor.process_messages(
        model=model,
        messages_list=messages_list,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Extract successful responses and handle failures
    responses = []
    successful_count = 0
    failed_count = 0

    for i, messages in enumerate(messages_list):
        # Find corresponding result
        found_result = None
        for success_result in result["successful"]:
            if success_result["messages"] == messages:
                found_result = success_result
                break

        if found_result:
            responses.append(found_result["data"])
            successful_count += 1
        else:
            # Find error result
            error_msg = "Unknown error"
            for failed_result in result["failed"]:
                if failed_result.get("messages") == messages:
                    error_msg = failed_result.get("error", "Unknown error")
                    break

            logger.error(f"Request {i} failed: {error_msg}")
            responses.append("")  # Empty string for failed requests
            failed_count += 1

    logger.info(
        f"Batch processing completed: {successful_count} successful, {failed_count} failed"
    )
    return responses


async def process_prompts_with_openai_async(
    model: str,
    prompts: List[str],
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 512,
    max_concurrent: int = 5,
    max_requests_per_second: int = 200,
) -> List[str]:
    """Process a list of prompts with OpenAI in parallel."""
    from .parallel_llm_utils import process_openai_requests_parallel

    result = await process_openai_requests_parallel(
        model=model,
        prompts=prompts,
        max_concurrent=max_concurrent,
        max_requests_per_second=max_requests_per_second,
        max_tokens=max_tokens,
        temperature=temperature,
        system_prompt=system_prompt,
    )

    # Extract responses in the same order as prompts
    responses = ["" for _ in prompts]

    # Map successful results back to their original positions (order is already maintained)
    for i, success_result in enumerate(result["successful"]):
        if success_result["success"]:
            responses[i] = success_result["data"]
        else:
            responses[i] = ""  # Failed request - return empty string

    # Log failed requests
    for failed_result in result["failed"]:
        prompt = failed_result.get("prompt", "Unknown prompt")
        error = failed_result.get("error", "Unknown error")
        logger.error(f"Failed request for prompt '{prompt[:50]}...': {error}")

    logger.info(f"Processed {len(prompts)} prompts: {result['stats']}")
    return responses
