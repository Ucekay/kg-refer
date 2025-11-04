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

logger = logging.getLogger(__name__)


class SchemaCanonicalizer:
    # The class to handle the last stage: Schema Canonicalization
    def __init__(
        self,
        target_schema_dict: dict,
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
        prompt_template_str: str,
        candidate_relation_definition_dict: dict,
        relation_example_dict: Optional[dict] = None,
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
            if relation_example_dict is not None:
                choices += f"Example: '{relation_example_dict[candidate_relations[idx]]['triple']}' can be extracted from '{candidate_relations[idx]['sentence']}'\n"
        choices += (
            f"{chr(ord('@') + len(candidate_relations) + 1)}. None of the above.\n"
        )

        verification_prompt = prompt_template_str.format_map(
            {
                "input_text": input_text_str,
                "query_triplet": query_triplet,
                "query_relation": query_triplet[1],
                "query_relation_definition": query_relation_definition,
                "choices": choices,
            }
        )

        messages = [{"role": "user", "content": verification_prompt}]
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
                self.verifier_openai_model, None, messages, max_tokens=50
            )

        if verification_result[0] in choice_letters_list:
            canonicalized_triplet[1] = candidate_relations[
                choice_letters_list.index(verification_result[0])
            ]
        else:
            return None

        return canonicalized_triplet

    async def llm_verify_async(
        self,
        input_text_str: str,
        query_triplet: List[str],
        query_relation_definition: str,
        prompt_template_str: str,
        candidate_relation_definition_dict: dict,
        relation_example_dict: Optional[dict] = None,
        max_concurrent: int = 5,
        max_requests_per_second: int = 200,
    ):
        """Async version of llm_verify for parallel processing."""
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
            if relation_example_dict is not None:
                choices += f"Example: '{relation_example_dict[candidate_relations[idx]]['triple']}' can be extracted from '{candidate_relations[idx]['sentence']}'\n"
        choices += (
            f"{chr(ord('@') + len(candidate_relations) + 1)}. None of the above.\n"
        )

        verification_prompt = prompt_template_str.format_map(
            {
                "input_text": input_text_str,
                "query_triplet": query_triplet,
                "query_relation": query_triplet[1],
                "query_relation_definition": query_relation_definition,
                "choices": choices,
            }
        )

        messages = [{"role": "user", "content": verification_prompt}]

        if self.verifier_openai_model is None:
            # For local models, fall back to synchronous processing
            verification_result = llm_utils.generate_completion_transformers(
                messages,
                self.verifier_model,
                self.verifier_tokenizer,
                answer_prepend="Answer: ",
                max_new_tokens=5,
            )
        else:
            # Use async OpenAI completion
            verification_result = await llm_utils.openai_chat_completion_async(
                self.verifier_openai_model,
                None,
                messages,
                max_tokens=50,
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
        open_triplet,
        open_relation_definition_dict: dict,
        verify_prompt_template: str,
        enrich=False,
    ):
        open_relation = open_triplet[1]

        if open_relation in self.schema_dict:
            # The relation is already canonical
            # candidate_relations, candidate_scores = self.retrieve_similar_relations(
            #     open_relation_definition_dict[open_relation]
            # )
            return open_triplet, {}

        candidate_relations = []
        candidate_scores = []

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
                    verify_prompt_template,
                    candidate_relations,
                    None,
                )
        else:
            canonicalized_triplet = None

        if canonicalized_triplet is None:
            # Cannot be canonicalized
            if enrich:
                self.schema_dict[open_relation] = open_relation_definition_dict[
                    open_relation
                ]
                if "sts_query" in self.embedder.prompts:
                    embedding = self.embedder.encode(
                        open_relation_definition_dict[open_relation],
                        prompt_name="sts_query",
                    )
                else:
                    embedding = self.embedder.encode(
                        open_relation_definition_dict[open_relation]
                    )
                self.schema_embedding_dict[open_relation] = embedding
                canonicalized_triplet = open_triplet
        return canonicalized_triplet, dict(zip(candidate_relations, candidate_scores))

    async def canonicalize_triplets_batch_async(
        self,
        verification_data: List[Dict[str, Any]],
        batch_size: int = 10,
        max_concurrent: int = 5,
        max_requests_per_second: int = 200,
        enrich: bool = False,
    ) -> List[Tuple[Optional[List[str]], Dict[str, float]]]:
        """
        Process multiple triplets in parallel for canonicalization.

        Args:
            verification_data: List of dicts containing verification information
            batch_size: Number of triplets to process in each batch
            max_concurrent: Maximum concurrent requests
            max_requests_per_second: Rate limit for requests
            enrich: Whether to enrich schema with new relations

        Returns:
            List of tuples (canonicalized_triplet, candidate_scores_dict)
        """
        results = []

        # Process in batches
        for i in range(0, len(verification_data), batch_size):
            batch = verification_data[i : i + batch_size]

            # Prepare verification tasks for this batch
            tasks = []
            batch_indices = []

            for j, data in enumerate(batch):
                # Handle already canonical items
                if data.get("already_canonical", False):
                    # Set results for already canonical items (matches sequential logic)
                    data["result"] = data["triplet"]
                    data["candidates_dict"] = {}
                    continue

                # Handle no candidates items
                if data.get("no_candidates", False):
                    # Set results for no candidates items (matches sequential logic)
                    if enrich:
                        # Add to schema like sequential version
                        open_relation = data["triplet"][1]
                        relation_def = data["relation_definition"]
                        self.schema_dict[open_relation] = relation_def
                        if "sts_query" in self.embedder.prompts:
                            embedding = self.embedder.encode(
                                relation_def, prompt_name="sts_query"
                            )
                        else:
                            embedding = self.embedder.encode(relation_def)
                        self.schema_embedding_dict[open_relation] = embedding
                    data["result"] = data["triplet"]
                    data["candidates_dict"] = {}
                    continue

                task = self.llm_verify_async(
                    input_text_str=data["input_text"],
                    query_triplet=data["triplet"],
                    query_relation_definition=data["relation_definition"],
                    prompt_template_str=data["prompt_template"],
                    candidate_relation_definition_dict=data["candidates"],
                    max_concurrent=max_concurrent,
                    max_requests_per_second=max_requests_per_second,
                )
                tasks.append(task)
                batch_indices.append(i + j)

            # Execute tasks in parallel with progress bar
            if tasks:
                logger.info(
                    f"Processing batch {i // batch_size + 1}/{(len(verification_data) + batch_size - 1) // batch_size}"
                )

                # Show progress bar for batch processing with index tracking
                indexed_tasks = [
                    (batch_indices[j], tasks[j]) for j in range(len(tasks))
                ]
                batch_results = [None] * len(tasks)  # Pre-allocate with None
                completed_count = 0
                for completed_task in tqdm_asyncio.as_completed(
                    [task for _, task in indexed_tasks],
                    total=len(tasks),
                    desc=f"Batch {i // batch_size + 1} - Verifying {len(tasks)} triplets",
                ):
                    result = await completed_task
                    # Find original batch index for this completed task
                    for original_batch_idx, original_task in indexed_tasks:
                        if original_task is completed_task:
                            batch_results[original_batch_idx] = result
                            break
                    completed_count += 1

                    # Log progress every few completions
                    if completed_count % 5 == 0 or completed_count == len(tasks):
                        logger.info(
                            f"Batch {i // batch_size + 1}: {completed_count}/{len(tasks)} verifications completed"
                        )

                # Process results
                for k, result in enumerate(batch_results):
                    original_idx = batch_indices[k]
                    if isinstance(result, Exception):
                        logger.error(
                            f"Verification failed for triplet {original_idx}: {result}"
                        )
                        # Fall back to original triplet
                        verification_data[original_idx]["result"] = verification_data[
                            original_idx
                        ]["triplet"]
                        verification_data[original_idx]["candidates_dict"] = {}
                    else:
                        if result is None:
                            # LLM returned None (cannot be canonicalized) - handle like sequential version
                            if enrich:
                                # Add to schema like sequential version
                                open_relation = verification_data[original_idx][
                                    "triplet"
                                ][1]
                                relation_def = verification_data[original_idx][
                                    "relation_definition"
                                ]
                                self.schema_dict[open_relation] = relation_def
                                if "sts_query" in self.embedder.prompts:
                                    embedding = self.embedder.encode(
                                        relation_def, prompt_name="sts_query"
                                    )
                                else:
                                    embedding = self.embedder.encode(relation_def)
                                self.schema_embedding_dict[open_relation] = embedding
                            verification_data[original_idx]["result"] = (
                                verification_data[original_idx]["triplet"]
                            )
                            verification_data[original_idx]["candidates_dict"] = {}
                        else:
                            # Successful verification
                            verification_data[original_idx]["result"] = result
                            verification_data[original_idx]["candidates_dict"] = (
                                verification_data[original_idx].get(
                                    "candidates_dict", {}
                                )
                            )

        # Compile final results in original order
        for data in verification_data:
            results.append((data["result"], data["candidates_dict"]))

        # Log final statistics
        total_triplets = len(verification_data)
        already_canonical = sum(
            1 for data in verification_data if data.get("already_canonical", False)
        )
        no_candidates = sum(
            1 for data in verification_data if data.get("no_candidates", False)
        )
        verified = total_triplets - already_canonical - no_candidates

        logger.info(f"Schema canonicalization statistics:")
        logger.info(f"  Total triplets: {total_triplets}")
        logger.info(f"  Already canonical: {already_canonical}")
        logger.info(f"  No candidates: {no_candidates}")
        logger.info(f"  Verified: {verified}")

        if verified > 0:
            successful_verifications = sum(
                1
                for data in verification_data
                if not data.get("already_canonical", False)
                and not data.get("no_candidates", False)
                and data.get("result") is not None
            )
            success_rate = (successful_verifications / verified) * 100
            logger.info(
                f"  Successful verifications: {successful_verifications}/{verified} ({success_rate:.1f}%)"
            )

        return results
