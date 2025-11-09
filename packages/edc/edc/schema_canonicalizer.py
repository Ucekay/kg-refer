import asyncio
import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from tqdm.asyncio import tqdm as tqdm_asyncio
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

import edc.utils.llm_utils as llm_utils
from edc.utils.async_utils import AsyncOpenAIProcessor

logger = logging.getLogger(__name__)


class SchemaCanonicalizer:
    # The class to handle the last stage: Schema Canonicalization
    def __init__(
        self,
        target_schema_dict: Dict[str, str],
        embedder: SentenceTransformer,
        verify_model: Optional[PreTrainedModel] = None,
        verify_tokenizer: Optional[PreTrainedTokenizerFast] = None,
        verify_openai_model: Optional[str] = None,
    ) -> None:
        # The canonicalizer uses an embedding model to first fetch candidates from the target schema, then uses a verifier schema to decide which one to canonicalize to or not
        # canonoicalize at all.

        assert verify_openai_model is not None or (
            verify_model is not None and verify_tokenizer is not None
        )
        self.verifier_model = verify_model
        self.verifier_tokenizer = verify_tokenizer
        self.verifier_openai_model = verify_openai_model
        self.schema_dict = target_schema_dict

        self.embedder = embedder

        # Embed the target schema
        self.schema_embedding_dict = {}

        print("Embedding target schema...")
        for relation, relation_definition in tqdm(target_schema_dict.items()):
            embedding = self.embedder.encode(relation_definition)
            self.schema_embedding_dict[relation] = embedding

    def retrieve_similar_relations(self, query_relation_definition: str, top_k=5):
        target_relation_list = list(self.schema_embedding_dict.keys())
        target_relation_embedding_list = list(self.schema_embedding_dict.values())
        if "sts_query" in self.embedder.prompts:
            query_embedding = self.embedder.encode(
                query_relation_definition, prompt_name="sts_query"
            )
        else:
            query_embedding = self.embedder.encode(query_relation_definition)

        scores = (
            np.array([query_embedding]) @ np.array(target_relation_embedding_list).T
        )

        scores = scores[0]
        highest_score_indices = np.argsort(-scores)

        return {
            target_relation_list[idx]: self.schema_dict[target_relation_list[idx]]
            for idx in highest_score_indices[:top_k]
        }, [scores[idx] for idx in highest_score_indices[:top_k]]

    def llm_verify(
        self,
        input_text_str: str,
        query_triplet: List[str],
        query_relation_definition: str,
        instructions_template_str: str,
        input_template_str: str,
        candidate_relation_definition_dict: Dict[str, str],
    ):
        canonicalized_triplet = copy.deepcopy(query_triplet)
        choice_letters_list = []
        choices = ""
        candidate_relations = list(candidate_relation_definition_dict.keys())
        candidate_relation_descriptions = list(
            candidate_relation_definition_dict.values()
        )
        for idx, rel in enumerate(candidate_relations):
            choice_letter = chr(ord("@") + idx + 1)
            choice_letters_list.append(choice_letter)
            choices += (
                f"{choice_letter}. '{rel}': {candidate_relation_descriptions[idx]}\n"
            )

        choices += (
            f"{chr(ord('@') + len(candidate_relations) + 1)}. None of the above.\n"
        )

        filled_instructions = instructions_template_str.format_map({})
        filled_input = input_template_str.format_map(
            {
                "input_text": input_text_str,
                "query_triplet": query_triplet,
                "query_relation": query_triplet[1],
                "query_relation_definition": query_relation_definition,
                "choices": choices,
            }
        )

        messages = [
            {"role": "system", "content": filled_instructions},
            {"role": "user", "content": filled_input},
        ]

        # For compatibility with existing code that expects input, instructions
        input, instructions = filled_input, filled_instructions

        if self.verifier_openai_model is None:
            # llm_utils.generate_completion_transformers([messages], self.model, self.tokenizer, device=self.device)
            verification_result = llm_utils.generate_completion_transformers(
                messages,
                self.verifier_model,
                self.verifier_tokenizer,
                answer_prepend="Answer: ",
                max_new_tokens=5,
            )
        else:
            verification_result = llm_utils.openai_chat_completion(
                self.verifier_openai_model, instructions, input, max_tokens=50
            )

        if verification_result[0] in choice_letters_list:
            canonicalized_triplet[1] = candidate_relations[
                choice_letters_list.index(verification_result[0])
            ]
        else:
            return None

        return canonicalized_triplet

    def canonicalize(
        self,
        input_text_str: str,
        open_triplet: List[str],
        open_relation_definition_dict: Dict[str, str],
        verify_instructions_template: str,
        verify_input_template: str,
        enrich=False,
    ):
        open_relation = open_triplet[1]

        if open_relation in self.schema_dict:
            # The relation is already canonical
            # candidate_relations, candidate_scores = self.retrieve_similar_relations(
            #     open_relation_definition_dict[open_relation]
            # )
            return open_triplet, {}

        candidate_relations: Dict[str, str] = {}
        candidate_scores: List[float] = []

        if len(self.schema_dict) != 0:
            if open_relation not in open_relation_definition_dict:
                canonicalized_triplet = None
            else:
                candidate_relations, candidate_scores = self.retrieve_similar_relations(
                    open_relation_definition_dict[open_relation]
                )
                canonicalized_triplet = self.llm_verify(
                    input_text_str,
                    open_triplet,
                    open_relation_definition_dict[open_relation],
                    verify_instructions_template,
                    verify_input_template,
                    candidate_relations,
                )
        else:
            canonicalized_triplet = None

        if canonicalized_triplet is None:
            # Cannot be canonicalized
            if enrich:
                # Use definition if available, otherwise use relation name as definition
                if open_relation in open_relation_definition_dict:
                    relation_definition = open_relation_definition_dict[open_relation]
                else:
                    relation_definition = (
                        open_relation  # Fallback: use relation name as definition
                    )

                self.schema_dict[open_relation] = relation_definition

                # Generate embedding for the relation definition
                if "sts_query" in self.embedder.prompts:
                    embedding = self.embedder.encode(
                        relation_definition,
                        prompt_name="sts_query",
                    )
                else:
                    embedding = self.embedder.encode(relation_definition)

                self.schema_embedding_dict[open_relation] = embedding
                canonicalized_triplet = open_triplet
        return canonicalized_triplet, dict(zip(candidate_relations, candidate_scores))


class OpenAIAsyncSchemaCanonicalizer:
    def __init__(
        self,
        target_schema_dict: Dict[str, str],
        embedder: SentenceTransformer,
        verify_model_name: str,
        max_concurrent: int = 200,
        max_req_per_sec: int = 80,
    ) -> None:
        self.verify_model_name = verify_model_name
        self.schema_dict = target_schema_dict

        self.embedder = embedder

        # Embed the target schema
        self.schema_embedding_dict = {}

        self.max_concurrent = max_concurrent
        self.max_req_per_sec = max_req_per_sec

        print("Embedding target schema...")
        for relation, relation_definition in tqdm(target_schema_dict.items()):
            embedding = self.embedder.encode(relation_definition)
            self.schema_embedding_dict[relation] = embedding

    def retrieve_similar_relations(self, query_relation_definition: str, top_k=5):
        target_relation_list = list(self.schema_embedding_dict.keys())
        target_relation_embedding_list = list(self.schema_embedding_dict.values())
        if "sts_query" in self.embedder.prompts:
            query_embedding = self.embedder.encode(
                query_relation_definition, prompt_name="sts_query"
            )
        else:
            query_embedding = self.embedder.encode(query_relation_definition)

        scores = (
            np.array([query_embedding]) @ np.array(target_relation_embedding_list).T
        )

        scores = scores[0]
        highest_score_indices = np.argsort(-scores)

        return {
            target_relation_list[idx]: self.schema_dict[target_relation_list[idx]]
            for idx in highest_score_indices[:top_k]
        }, [scores[idx] for idx in highest_score_indices[:top_k]]

    async def llm_verify_async(
        self,
        input_text: str,
        query_triplet_list: List[List[str]],
        query_relation_definition_dict: Dict[str, str],
        instructions_template: str,
        input_template_str: str,
        candidate_relation_definition_dict_list: List[Dict[str, str]],
        enrich=False,
    ):
        async def process_single(
            query_triplet: List[str],
            candidate_relations: List[str],
            choice_letters_list: List[str],
            openai_async_processor: AsyncOpenAIProcessor,
            model_name: str,
            instructions: str,
            input: str | None,
            max_tokens: int,
            enrich=False,
        ):
            if input is None:
                if query_triplet[1] in self.schema_dict:
                    return query_triplet
                else:
                    return None
            else:
                responses = await openai_async_processor.openai_responses_async(
                    model_name,
                    instructions,
                    input,
                    max_tokens=max_tokens,
                )
                if responses in choice_letters_list:
                    query_triplet[1] = candidate_relations[
                        choice_letters_list.index(responses)
                    ]
                    return query_triplet
                else:
                    return None

        filled_input_list: List[str | None] = []
        choice_letters_list_list = []
        candidate_relations_list = []
        for i, query_triplet in enumerate(query_triplet_list):
            if (
                len(self.schema_dict) == 0
                or query_triplet[1] in self.schema_dict
                or query_triplet[1] not in query_relation_definition_dict
            ):
                query_relation_definition = None
                filled_input = None
                choice_letters_list = []
                candidate_relations = []

            else:
                query_relation_definition = query_relation_definition_dict[
                    query_triplet[1]
                ]
                candidate_relation_definition_dict = (
                    candidate_relation_definition_dict_list[i]
                )

                choice_letters_list = []
                choices = ""
                candidate_relations = list(candidate_relation_definition_dict.keys())
                candidate_relation_descriptions = list(
                    candidate_relation_definition_dict.values()
                )
                for j, rel in enumerate(candidate_relations):
                    choice_letter = chr(ord("@") + j + 1)
                    choice_letters_list.append(choice_letter)
                    choices += f"{choice_letter}. '{rel}': {candidate_relation_descriptions[j]}\n"

                choices += f"{chr(ord('@') + len(candidate_relations) + 1)}. None of the above.\n"

                filled_input = input_template_str.format_map(
                    {
                        "input_text": input_text,
                        "query_triplet": query_triplet,
                        "query_relation": query_triplet[1],
                        "query_relation_definition": query_relation_definition,
                        "choices": choices,
                    }
                )
            filled_input_list.append(filled_input)
            choice_letters_list_list.append(choice_letters_list)
            candidate_relations_list.append(candidate_relations)

        opneai_async_processor = AsyncOpenAIProcessor(
            max_concurrent=self.max_concurrent, max_req_per_sec=self.max_req_per_sec
        )

        tasks = [
            process_single(
                query_triplet,
                candidate_relations,
                choice_letters_list,
                opneai_async_processor,
                self.verify_model_name,
                instructions_template,
                filled_input,
                max_tokens=50,
                enrich=enrich,
            )
            for filled_input, query_triplet, candidate_relations, choice_letters_list in zip(
                filled_input_list,
                query_triplet_list,
                candidate_relations_list,
                choice_letters_list_list,
            )
        ]

        results: List[List[str] | None] = await asyncio.gather(*tasks)

        return results

    async def canonicalize_async(
        self,
        input_text: str,
        open_triplets: List[List[str]],
        open_relation_definition_dict: Dict[str, str],
        instructions_template: str,
        input_template: str,
        enrich=False,
    ):
        candidate_relation_definition_dict_list: List[Dict[str, str]] = []
        candidate_relations_and_scores_dict_list = []

        # Track indices for different conditions
        schema_existing_indices = []  # Relations already in schema_dict
        empty_schema_or_missing_def_indices = []  # Empty schema or missing definitions

        for i, open_triplet in enumerate(open_triplets):
            open_relation = open_triplet[1]

            if open_relation in self.schema_dict:
                schema_existing_indices.append(i)
                candidate_relation_definition_dict_list.append({})
                candidate_relations_and_scores_dict_list.append({})
                continue

            if (
                len(self.schema_dict) == 0
                or open_relation not in open_relation_definition_dict
            ):
                empty_schema_or_missing_def_indices.append(i)
                candidate_relation_definition_dict_list.append({})
                candidate_relations_and_scores_dict_list.append({})
                continue

            candidate_relation_definition_dict, candidate_scores = (
                self.retrieve_similar_relations(
                    open_relation_definition_dict[open_relation]
                )
            )
            candidate_relation_definition_dict_list.append(
                candidate_relation_definition_dict
            )
            candidate_relations_and_scores_dict_list.append(
                dict(
                    zip(
                        candidate_relation_definition_dict.keys(),
                        candidate_scores,
                    )
                )
            )

        canonicalized_triplet_list = await self.llm_verify_async(
            input_text,
            open_triplets,
            open_relation_definition_dict,
            instructions_template,
            input_template,
            candidate_relation_definition_dict_list,
            enrich=enrich,
        )

        for idx, triplet in enumerate(canonicalized_triplet_list):
            if triplet is None:
                if enrich:
                    # Add relation to schema even if not in definition dict
                    relation_name = open_triplets[idx][1]

                    # Use definition if available, otherwise use relation name as definition
                    if relation_name in open_relation_definition_dict:
                        relation_definition = open_relation_definition_dict[
                            relation_name
                        ]
                    else:
                        logger.warning(
                            f"Enriching schema with relation '{relation_name}' without definition."
                        )
                        relation_definition = (
                            relation_name  # Fallback: use relation name as definition
                        )

                    self.schema_dict[relation_name] = relation_definition

                    # Generate embedding for the relation definition
                    if "sts_query" in self.embedder.prompts:
                        embedding = self.embedder.encode(
                            relation_definition,
                            prompt_name="sts_query",
                        )
                    else:
                        embedding = self.embedder.encode(relation_definition)

                    self.schema_embedding_dict[relation_name] = embedding
                    canonicalized_triplet_list[idx] = open_triplets[idx]

        return canonicalized_triplet_list, candidate_relations_and_scores_dict_list

    async def canonicalize_all_async(
        self,
        input_text_list: List[str],
        open_triplets_list: List[List[List[str]]],
        open_relation_definition_dict_list: List[Dict[str, str]],
        instructions_template: str,
        input_template: str,
        enrich=False,
    ):
        tasks = [
            self.canonicalize_async(
                input_text,
                open_triplets,
                open_relation_definition_dict,
                instructions_template,
                input_template,
                enrich,
            )
            for input_text, open_triplets, open_relation_definition_dict in zip(
                input_text_list,
                open_triplets_list,
                open_relation_definition_dict_list,
            )
        ]
        results: List[
            Tuple[List[List[str] | None], List[Dict[str, float]]]
        ] = await tqdm_asyncio.gather(*tasks, desc="Canonicalizing all inputs")

        # Split results into separate lists
        canonicalized_triplet_list_list = [result[0] for result in results]
        candidate_relations_and_scores_dict_list_list = [
            result[1] for result in results
        ]

        return (
            canonicalized_triplet_list_list,
            candidate_relations_and_scores_dict_list_list,
        )
