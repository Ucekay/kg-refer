import ast
import gc
import logging
import os
import time
from typing import Dict, List, Optional

import openai
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


def parse_raw_entities(raw_entities: str):
    parsed_entities = []
    try:
        left_bracket_index = raw_entities.index("[")
        right_bracket_index = raw_entities.rindex("]") + 1
        try:
            parsed_entities: List[str] = ast.literal_eval(
                raw_entities[left_bracket_index : right_bracket_index + 1]
            )
        except Exception as e:
            logging.debug(f"Failed to parse entities from {raw_entities}: {e}")
    except ValueError:
        logging.debug(f"No brackets found in entities string: {raw_entities}")
    logging.debug(f"Entities {raw_entities} parsed as {parsed_entities}")
    return parsed_entities


def parse_raw_triplets(raw_triplets: str) -> List[List[str]]:
    unmatched_left_bracket_indices = []
    matched_bracket_pairs = []

    collected_triplets = []
    for c_idx, c in enumerate(raw_triplets):
        if c == "[":
            unmatched_left_bracket_indices.append(c_idx)
        if c == "]":
            if len(unmatched_left_bracket_indices) == 0:
                continue
            matched_left_bracket_idx = unmatched_left_bracket_indices.pop()
            matched_bracket_pairs.append((matched_left_bracket_idx, c_idx))

    for l, r in matched_bracket_pairs:
        bracketed_str = raw_triplets[l : r + 1]
        try:
            parsed_triplet: List[str] = ast.literal_eval(bracketed_str)
            if len(parsed_triplet) == 3 and all(
                isinstance(t, str) for t in parsed_triplet
            ):
                if all([e != "" and e != "_" for e in parsed_triplet]):
                    collected_triplets.append(parsed_triplet)
            elif not all([type(x) == type(parsed_triplet[0]) for x in parsed_triplet]):
                for e_idx, e in enumerate(parsed_triplet):
                    if isinstance(e, list):
                        parsed_triplet[e_idx] = ", ".join(e)
                collected_triplets.append(parsed_triplet)
        except Exception as e:
            pass
    logger.debug(f"Triplets {raw_triplets} parsed as {collected_triplets}")
    return collected_triplets


def parse_relation_definition(raw_definitions: str):
    descriptions = raw_definitions.split("\n")
    relation_definition_dict: dict[str, str] = {}

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
        f"Relation Definitions {raw_definitions} parsed as {relation_definition_dict}"
    )
    return relation_definition_dict


def is_model_openai(model_name: str) -> bool:
    return "gpt" in model_name.lower()


def generate_completion_transformers(
    input: List[Dict[str, str]],
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
    instructions: str,
    input: str,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = None

    while response is None:
        try:
            response = client.responses.create(
                model=model,
                instructions=instructions,
                input=input,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

        except openai.APIStatusError as e:
            logging.warning(f"OpenAI API request failed with error: {e}. Retrying...")
            time.sleep(60)
        except openai.APIError as e:
            logging.warning(f"OpenAI API request failed with error: {e}. Retrying...")
            time.sleep(1)
        except Exception as e:
            logging.warning(f"OpenAI API request failed with error: {e}. Retrying...")
            time.sleep(1)

        # レスポンス処理
    result = response.output_text
    logging.debug(
        f"Model: {model}\nInput: {input}\nInstructions: {instructions}\nResult: {result}"
    )
    return result


def convert_to_responses_format(prompt: str):
    return _split_prompt(prompt)


def _split_prompt(prompt: str):
    """
    INPUT_DATA_START/ENDで区切られたプロンプトをinstructionsとinputに分離
    """

    parts = prompt.split("INPUT_DATA_START")
    instructions_part = parts[0].strip()
    input_part = parts[1].replace("INPUT_DATA_END", "").strip()
    return input_part, instructions_part


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
