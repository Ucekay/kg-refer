"""Parallel processing utilities for EDC framework.

This module provides parallel processing functions that are completely independent
from the main EDC class to avoid CUDA multiprocessing issues.
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def extract_entities_and_relations_from_triplets(
    triplets_list: List[List[List[str]]],
) -> List[Tuple[List[str], List[str]]]:
    """Extract entities and relations from triplets using multiprocessing.

    This function is completely independent from EDC class to avoid CUDA
    resource handle issues when using ProcessPoolExecutor.

    Args:
        triplets_list: List of triplet lists, where each triplet is [entity, relation, entity]

    Returns:
        List of tuples containing (entities, relations) for each triplet list
    """
    # Set environment variable to avoid tokenizers parallelism warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    results = []

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(_process_single_triplets_helper, triplets_list))

    return results


def _process_single_triplets_helper(
    triplet_list: List[List[str]],
) -> Tuple[List[str], List[str]]:
    """Process a single list of triplets to extract unique entities and relations.

    Args:
        triplet_list: List of triplets, where each triplet is [entity, relation, entity]

    Returns:
        Tuple of (unique_entities, unique_relations)
    """
    try:
        arr = np.array(triplet_list)

        if arr.ndim != 2 or arr.shape[1] != 3:
            return [], []

        entities = np.unique(arr[:, [0, 2]].flatten()).tolist()
        relations = np.unique(arr[:, 1]).tolist()

        return entities, relations

    except Exception as e:
        logger.error(f"Error processing triplets {triplet_list}: {e}")
        return [], []
