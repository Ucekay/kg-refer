import copy
import csv
import json
import logging
import pathlib
import random
from importlib import reload
from typing import List, Literal, Optional, TypedDict, Union

import torch
from mpmath.libmp.backend import os
from sentence_transformers import SentenceTransformer
from torch.fx.experimental.symbolic_shapes import canonicalize_bool_expr
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

from edc import schema_canonicalizer
from edc.entity_extractor import LocalEntityExtractor, OpenAIEntityExtractor
from edc.extractor import LocalExtractor, OpenAIExtractor
from edc.schema_canonicalizer import SchemaCanonicalizer
from edc.schema_definer import LocalSchemaDefiner, OpenAISchemaDefiner
from edc.schema_retriever import SchemaRetriever

from .types.config import EDCConfig
from .utils import llm_utils


class RelationExample(TypedDict):
    text: str
    triplet: List[str]


reload(logging)
logger = logging.getLogger(__name__)


class EDC:
    schema: dict[str, str]

    def __init__(self, config: EDCConfig) -> None:
        """Initialize EDC framework with optional configuration.

        Args:
            config: Configuration object containing EDC settings
        """
        self.oie_llm_name = config.oie_llm
        self.oie_prompt_template_file_path = config.oie_prompt_template_file_path
        self.oie_few_shot_example_file_path = config.oie_few_shot_example_file_path

        self.sd_llm_name = config.sd_llm
        self.sd_prompt_template_file_path = config.sd_prompt_template_file_path
        self.sd_few_shot_example_file_path = config.sd_few_shot_example_file_path

        self.sc_llm_name = config.sc_llm
        self.sc_embedder_name = config.sc_embedder
        self.sc_prompt_template_file_path = config.sc_prompt_template_file_path

        self.sr_adapter_path = config.sr_adapter_path
        self.sr_embedder_name = config.sr_embedder
        self.r_oie_prompt_template_file_path = (
            config.refined_oie_prompt_template_file_path
        )
        self.r_oie_few_shot_example_file_path = (
            config.refined_oie_few_shot_example_file_path
        )

        self.ee_llm_name = config.ee_llm
        self.ee_prompt_template_file_path = config.ee_prompt_template_file_path
        self.ee_few_shot_example_file_path = config.ee_few_shot_example_file_path

        self.em_prompt_template_file_path = config.em_prompt_template_file_path

        self.initial_schema_path = config.target_schema_path
        self.enrich_chema = config.enrich_schema

        if self.initial_schema_path.strip():
            reader = csv.reader(open(self.initial_schema_path, "r", encoding="utf-8"))
            self.schema = {
                relation: relation_definition
                for relation, relation_definition in reader
            }

        else:
            self.schema = {}

        self.needed_model_set = set(
            [
                self.oie_llm_name,
                self.sd_llm_name,
                self.sc_llm_name,
                self.sc_embedder_name,
                self.ee_llm_name,
            ]
        )

        self.loaded_hf_model_dict: dict[
            str, tuple[PreTrainedModel, PreTrainedTokenizerFast]
        ] = {}

        self.loaded_sts_model_dict: dict[str, SentenceTransformer] = {}
        logging.basicConfig(level=getattr(logging, config.log_level.upper()))

        logger.info(f"Model used: {self.needed_model_set}")

    def oie(
        self,
        input_text_list: List[str],
        previous_extractex_triplets_list: Optional[List[List[str]]] = None,
        free_model: bool = False,
    ):
        oie_model = None
        oie_tokenizer = None

        if not llm_utils.is_model_openai(self.oie_llm_name):
            oie_model, oie_tokenizer = self.load_hf_model(self.oie_llm_name)
            extractor = LocalExtractor(oie_model, oie_tokenizer)
        else:
            extractor = OpenAIExtractor(self.oie_llm_name)

        oie_triplets_list = []
        entity_hint_list = None
        relation_hint_list = None

        if previous_extractex_triplets_list is not None:
            logger.info("Running Refined OIE...")
            oie_refinement_prompt_template_str = open(
                self.r_oie_prompt_template_file_path, encoding="utf-8"
            ).read()
            oie_refinement_few_shot_example_str = open(
                self.r_oie_few_shot_example_file_path, encoding="utf-8"
            ).read()

            logger.info("Putting together the refinement hint...")
            entity_hint_list, relation_hint_list = self.construct_refinement_hint(
                input_text_list, previous_extractex_triplets_list, free_model=free_model
            )

            assert len(previous_extractex_triplets_list) == len(input_text_list)
            for idx, input_text in enumerate(tqdm(input_text_list)):
                input_text = input_text_list[idx]
                entity_hint_str = entity_hint_list[idx]
                relation_hint_str = relation_hint_list[idx]
                refined_oie_triplets = extractor.extract(
                    input_text,
                    oie_refinement_few_shot_example_str,
                    oie_refinement_prompt_template_str,
                    entity_hint_str,
                    relation_hint_str,
                )
                oie_triplets_list.append(refined_oie_triplets)
        else:
            entity_hint_list = ["" for _ in input_text_list]
            relation_hint_list = ["" for _ in input_text_list]
            logger.info("Running OIE...")
            oie_few_shot_example_str = open(
                self.oie_few_shot_example_file_path, encoding="utf-8"
            ).read()
            oie_prompt_template_str = open(
                self.oie_prompt_template_file_path, encoding="utf-8"
            ).read()

            for input_text in tqdm(input_text_list):
                oie_triplets = extractor.extract(
                    input_text,
                    oie_few_shot_example_str,
                    oie_prompt_template_str,
                )
                oie_triplets_list.append(oie_triplets)
                logger.debug(f"{input_text}\n -> {oie_triplets}\n")

        logger.info("OIE finished.")

        if free_model:
            logger.info(f"Freeing model {self.oie_llm_name} as it is no longer needed.")
            llm_utils.free_model(oie_model, oie_tokenizer)
            del self.loaded_hf_model_dict[self.oie_llm_name]

        return oie_triplets_list, entity_hint_list, relation_hint_list

    def load_hf_model(
        self, model_name: str
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerFast]:
        if model_name in self.loaded_hf_model_dict:
            logger.info(f"Model {model_name} is already loaded, reusing it.")
        else:
            logger.info(f"Loading model {model_name}...")
            model, tokenizer = (
                AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    dtype=torch.bfloat16,
                ),
                AutoTokenizer.from_pretrained(model_name),
            )
            self.loaded_hf_model_dict[model_name] = (model, tokenizer)

        return self.loaded_hf_model_dict[model_name]

    def load_sts_model(self, model_name: str) -> SentenceTransformer:
        if model_name in self.loaded_sts_model_dict:
            logger.info(f"Model {model_name} is already loaded, reusing it.")
        else:
            logger.info(f"Loading model {model_name}...")
            model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": torch.bfloat16},
            )
            self.loaded_sts_model_dict[model_name] = model

        return self.loaded_sts_model_dict[model_name]

    def schema_definition(
        self,
        input_text_list: List[str],
        oie_triplets_list: List[List[str]],
        free_model=False,
    ):
        assert len(input_text_list) == len(oie_triplets_list)

        sd_model = None
        sd_tokenizer = None

        if not llm_utils.is_model_openai(self.sd_llm_name):
            sd_model, sd_tokenizer = self.load_hf_model(self.sd_llm_name)
            schema_definer = LocalSchemaDefiner(sd_model, sd_tokenizer)
        else:
            schema_definer = OpenAISchemaDefiner(self.sd_llm_name)

        schema_definition_few_shot_examples_str = open(
            self.sd_few_shot_example_file_path, encoding="utf-8"
        ).read()
        schema_definition_prompt_template_str = open(
            self.sd_prompt_template_file_path, encoding="utf-8"
        ).read()
        schema_definition_dict_list = []

        logger.info("Running Schema Definition...")
        for idx, oie_triplets in enumerate(tqdm(oie_triplets_list)):
            schema_definition_dict = schema_definer.define_schema(
                input_text_list[idx],
                oie_triplets,
                schema_definition_few_shot_examples_str,
                schema_definition_prompt_template_str,
            )
            schema_definition_dict_list.append(schema_definition_dict)
            logger.debug(
                f"{input_text_list[idx]}, {oie_triplets}\n -> {schema_definition_dict}\n"
            )
        logger.info("Schema Definition finished.")
        if free_model:
            logger.info(f"Freeing model {self.sd_llm_name} as it is no longer needed.")
            llm_utils.free_model(sd_model, sd_tokenizer)
            del self.loaded_hf_model_dict[self.sd_llm_name]
        return schema_definition_dict_list

    def schema_canonicalization(
        self,
        input_text_list: List[str],
        oie_triplets_list: List[List[str]],
        schema_definition_dict_list: List[dict],
        free_model=False,
    ):
        assert len(input_text_list) == len(oie_triplets_list) and len(
            input_text_list
        ) == len(schema_definition_dict_list)
        logger.info("Running Schema Canonicalization...")

        sc_verify_prompt_template_str = open(
            self.sc_prompt_template_file_path, encoding="utf-8"
        ).read()

        sc_embedder = self.load_sts_model(self.sc_embedder_name)

        sc_verify_model = None
        sc_verify_tokenizer = None

        if not llm_utils.is_model_openai(self.sc_llm_name):
            sc_verify_model, sc_verify_tokenizer = self.load_hf_model(self.sc_llm_name)
            schema_canonicalizer = SchemaCanonicalizer(
                self.schema, sc_embedder, sc_verify_model, sc_verify_tokenizer
            )
        else:
            schema_canonicalizer = SchemaCanonicalizer(
                self.schema, sc_embedder, verify_openai_model=self.sc_llm_name
            )

        canonicalize_triplets_list = []
        canon_candidate_dict_per_entity_list = []

        for idx, input_text in enumerate(tqdm(input_text_list)):
            oie_triplets = oie_triplets_list[idx]
            canonicalized_triplets = []
            sd_dict = schema_definition_dict_list[idx]
            canon_candidate_dict_list = []
            for oie_triplet in oie_triplets:
                result = schema_canonicalizer.canonicalize(
                    input_text,
                    oie_triplet,
                    sd_dict,
                    sc_verify_prompt_template_str,
                    self.enrich_chema,
                )
                if result is not None:
                    canonicalized_triplet, canon_candidate_dict = result
                else:
                    canonicalized_triplet = oie_triplet
                    canon_candidate_dict = {}
                canonicalized_triplets.append(canonicalized_triplet)
                canon_candidate_dict_list.append(canon_candidate_dict)

            canonicalize_triplets_list.append(canonicalized_triplets)
            canon_candidate_dict_per_entity_list.append(canon_candidate_dict_list)

            logger.debug(
                f"{input_text}\n, {oie_triplets} ->\n {canonicalized_triplets}"
            )
            logger.debug(
                f"Retrieved candidate relations {canon_candidate_dict_list[-1] if canon_candidate_dict_list else {}}"
            )

        logger.info("Schema Canonicalization finished.")

        if free_model:
            logger.info(
                f"Freeing model {self.sc_llm_name, self.sc_embedder_name} as it is no longer needed."
            )
            if not llm_utils.is_model_openai(self.sc_llm_name):
                llm_utils.free_model(sc_verify_model, sc_verify_tokenizer)
                del self.loaded_hf_model_dict[self.sc_llm_name]

        return canonicalize_triplets_list, canon_candidate_dict_per_entity_list

    def construct_refinement_hint(
        self,
        input_text_list: List[str],
        extracted_triplets_list: List[List[List[str]]],
        include_relation_example="self",
        relation_top_k=10,
        free_model=False,
    ):
        entity_extraction_few_shot_example_str = open(
            self.ee_few_shot_example_file_path, encoding="utf-8"
        ).read()
        entity_extraction_prompt_template_str = open(
            self.ee_prompt_template_file_path, encoding="utf-8"
        ).read()

        entity_merging_prompt_template_str = open(
            self.em_prompt_template_file_path, encoding="utf-8"
        ).read()

        entity_hint_list = []
        relation_hint_list = []

        ee_model = None
        ee_tokenizer = None

        if not llm_utils.is_model_openai(self.ee_llm_name):
            ee_model, ee_tokenizer = self.load_hf_model(self.ee_llm_name)
            entity_extractor = LocalEntityExtractor(ee_model, ee_tokenizer)
        else:
            entity_extractor = OpenAIEntityExtractor(self.ee_llm_name)

        sr_embedding_model = self.load_sts_model(self.sr_embedder_name)

        schema_retriever = SchemaRetriever(
            self.schema, sr_embedding_model, None, finetuned_e5mistral=False
        )

        relation_example_dict: dict[str, List[RelationExample]] = {}
        if include_relation_example == "self":
            for idx in range(len(input_text_list)):
                input_text_str = input_text_list[idx]
                ectracted_triplets = extracted_triplets_list[idx]
                for triplet in ectracted_triplets:
                    relation = triplet[1]
                    if relation not in relation_example_dict:
                        relation_example_dict[relation] = [
                            {"text": input_text_str, "triplet": triplet}
                        ]
                    else:
                        relation_example_dict[relation].append(
                            {"text": input_text_str, "triplet": triplet}
                        )
        else:
            # Todo: allow to pass gold examples of relations
            pass

        for idx in tqdm(range(len(input_text_list))):
            input_text_str = input_text_list[idx]
            extracted_triplets = extracted_triplets_list[idx]

            previous_relations = set()
            previous_entities = set()

            for triplet in extracted_triplets:
                previous_relations.add(triplet[1])
                previous_entities.add(triplet[0])
                previous_entities.add(triplet[2])

            previous_entities = list(previous_entities)
            previous_relations = list(previous_relations)

            extracted_entities = entity_extractor.extract_entities(
                input_text_str,
                entity_extraction_few_shot_example_str,
                entity_extraction_prompt_template_str,
            )
            marge_entities = entity_extractor.merge_entities(
                input_text_str,
                previous_entities,
                extracted_entities,
                entity_merging_prompt_template_str,
            )
            entity_hint_list.append(str(marge_entities))

            hint_relations = previous_relations

            retrieved_relations = schema_retriever.retrieve_relevant_relations(
                input_text_str
            )

            counter = 0

            for relation in retrieved_relations:
                if counter >= relation_top_k:
                    break
                else:
                    if relation not in hint_relations:
                        hint_relations.append(relation)

            candidate_relation_str = ""
            for relation_idx, relation in enumerate(hint_relations):
                if relation not in self.schema:
                    continue

                relation_definition = self.schema[relation]

                candidate_relation_str += (
                    f"{relation_idx + 1}. {relation}: {relation_definition}\n"
                )
                if include_relation_example == "self":
                    if relation not in relation_example_dict:
                        pass
                    else:
                        selected_example = None
                        if len(relation_example_dict[relation]) != 0:
                            selected_example = random.choice(
                                relation_example_dict[relation]
                            )

                        if selected_example is not None:
                            candidate_relation_str += f"""For example, {selected_example["triplet"]} can be extracted from "{selected_example["text"]}"\n"""
                        else:
                            pass

            relation_hint_list.append(candidate_relation_str)

        if free_model:
            logger.info(
                f"Freeing model {self.sr_embedder_name, self.ee_llm_name} as it is no longer needed."
            )
            llm_utils.free_model(sr_embedding_model)
            llm_utils.free_model(ee_model, ee_tokenizer)
            del self.loaded_sts_model_dict[self.sr_embedder_name]
            del self.loaded_hf_model_dict[self.ee_llm_name]
        return entity_hint_list, relation_hint_list

    def extract_kg(
        self, input_text_list: List[str], output_dir: str, refinement_iterations=0
    ):
        if os.path.exists(output_dir):
            logger.warning(f"Output directory {output_dir} already exists.")
            exit()
        for iteration in range(refinement_iterations + 1):
            pathlib.Path(f"{output_dir}/iter{iteration}").mkdir(
                parents=True, exist_ok=True
            )

        logger.info("EDC starts running...")

        required_model_dics = {
            "oie": self.oie_llm_name,
            "sd": self.sd_llm_name,
            "sc_varify": self.sc_llm_name,
            "sc_embedd": self.sc_embedder_name,
            "ee": self.ee_llm_name,
            "sr": self.sr_embedder_name,
        }

        triplets_from_last_iteration = None
        for iteration in range(refinement_iterations + 1):
            logger.info(f"Iteration {iteration}:")

            iteration_result_dir = f"{output_dir}/iter{iteration}"

            required_model_dict_current_iteration = copy.deepcopy(required_model_dics)

            del required_model_dict_current_iteration["oie"]
            oie_triplets_list, entity_hint_list, relation_hint_list = self.oie(
                input_text_list,
                free_model=self.oie_llm_name
                not in required_model_dict_current_iteration.values()
                and iteration == refinement_iterations,
                previous_extractex_triplets_list=triplets_from_last_iteration,
            )

            del required_model_dict_current_iteration["sd"]
            sd_dict_list = self.schema_definition(
                input_text_list,
                oie_triplets_list,
                free_model=self.sd_llm_name
                not in required_model_dict_current_iteration.values()
                and iteration == refinement_iterations,
            )

            del required_model_dict_current_iteration["sc_varify"]
            del required_model_dict_current_iteration["sc_embedd"]
            canon_triplets_list, canon_candidate_dict_list = (
                self.schema_canonicalization(
                    input_text_list,
                    oie_triplets_list,
                    sd_dict_list,
                    free_model=self.sc_llm_name
                    not in required_model_dict_current_iteration.values()
                    and iteration == refinement_iterations,
                )
            )

            non_null_triplets_list = [
                [triplet for triplet in triplets if triplet is not None]
                for triplets in canon_triplets_list
            ]

            triplets_from_last_iteration = non_null_triplets_list

            assert len(oie_triplets_list) == len(sd_dict_list) and len(
                canon_triplets_list
            ) == len(sd_dict_list)

            json_results_list = []
            for idx in range(len(oie_triplets_list)):
                result_json = {
                    "index": idx,
                    "input_text": input_text_list[idx],
                    "entity_hint": entity_hint_list[idx],
                    "relation_hint": relation_hint_list[idx],
                    "oie": oie_triplets_list[idx],
                    "schema_definition": sd_dict_list[idx],
                    "canonicalization_candidates": str(canon_candidate_dict_list[idx]),
                    "schema_canonicalizaiton": canon_triplets_list[idx],
                }
                json_results_list.append(result_json)
            restult_at_each_stage_file = open(
                f"{iteration_result_dir}/result_at_each_stage.json", "w"
            )
            json.dump(
                json_results_list,
                restult_at_each_stage_file,
                indent=4,
            )

            final_result_file = open(f"{iteration_result_dir}/canon_kg.txt", "w")
            for idx, canon_triplets in enumerate(non_null_triplets_list):
                final_result_file.write(str(canon_triplets))
                if idx != len(canon_triplets_list) - 1:
                    final_result_file.write("\n")
                final_result_file.flush()

        return canon_triplets_list
