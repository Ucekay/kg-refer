"""Main explainer class for KGAT model path-based explanations."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from kgat_explainer.attention import AttentionCalculator
from kgat_explainer.path_finder import PathFinder
from kgat_explainer.predictor import KGATPredictor


class KGATExplainer:
    """Explains KGAT predictions by finding paths and computing attention scores."""

    def __init__(
        self,
        model: nn.Module,
        data_loader,
        kg_dict: Dict[int, List[Tuple[int, int]]],
        kg_dict_by_relation: Dict[int, List[Tuple[int, int]]],
        device: torch.device = torch.device("cpu"),
        precompute_attention: bool = True,
    ):
        """Initialize the explainer.

        Args:
            model: Trained KGAT model
            data_loader: KGAT DataLoader with train/val/test data
            kg_dict: Knowledge graph dictionary mapping head -> [(tail, relation), ...]
            kg_dict_by_relation: KG dictionary mapping relation -> [(head, tail), ...]
            device: Device to run computations on
            precompute_attention: Whether to precompute all attention scores
        """
        self.model = model
        self.data_loader = data_loader
        self.kg_dict = kg_dict
        self.kg_dict_by_relation = kg_dict_by_relation
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        self.n_entities = data_loader.n_entities
        self.device = device

        # Initialize components
        self.predictor = KGATPredictor(model, data_loader, device)

        self.attention_calculator = AttentionCalculator(
            model.entity_user_embed,
            model.relation_embed,
            model.trans_M,
            device,
        )

        self.path_finder = PathFinder(kg_dict, self.n_entities)

        # Precompute attention scores if requested
        self.attention_scores = {}
        if precompute_attention:
            print("Precomputing attention scores...")
            raw_scores = (
                self.attention_calculator.precompute_attention_scores_by_relation(
                    kg_dict_by_relation
                )
            )
            print("Normalizing attention scores...")
            self.attention_scores = self._normalize_scores(raw_scores)
            print(f"Precomputed and normalized {len(self.attention_scores)} attention scores")

    def _normalize_scores(
        self, raw_scores: Dict[Tuple[int, int, int], float]
    ) -> Dict[Tuple[int, int, int], float]:
        """Softmax normalize scores grouped by head entity.

        Args:
            raw_scores: Dictionary mapping (h, r, t) -> raw_attention_score

        Returns:
            Dictionary with normalized scores
        """
        import math
        from collections import defaultdict

        # 1. Calculate sum of exp(score) for each head (denominator)
        head_exp_sums = defaultdict(float)
        
        # For numerical stability, we could subtract max score per head,
        # but direct exp is usually fine for attention scores which are usually small.
        for (h, _, _), score in raw_scores.items():
            head_exp_sums[h] += math.exp(score)

        # 2. Normalize
        normalized_scores = {}
        for (h, r, t), score in raw_scores.items():
            denom = head_exp_sums[h]
            if denom > 0:
                normalized_scores[(h, r, t)] = math.exp(score) / denom
            else:
                normalized_scores[(h, r, t)] = 0.0

        return normalized_scores

    def get_attention_score(self, h: int, r: int, t: int) -> float:
        """Get attention score for an edge.

        Args:
            h: Head entity/user ID
            r: Relation ID
            t: Tail entity/user ID

        Returns:
            Attention score
        """
        if (h, r, t) in self.attention_scores:
            return self.attention_scores[(h, r, t)]
        else:
            # Compute on-the-fly if not precomputed
            return self.attention_calculator.calculate_edge_attention(h, t, r)

    def explain_prediction(
        self,
        user_id: int,
        item_id: int,
        max_hops: int = 3,
        max_paths: int = 10,
    ) -> Dict:
        """Explain a prediction by finding paths and their attention scores.

        Args:
            user_id: User ID (adjusted with entity offset)
            item_id: Item ID
            max_hops: Maximum number of hops to search
            max_paths: Maximum number of paths to return

        Returns:
            Dictionary containing paths and their scores
        """
        # Find paths
        paths = self.path_finder.find_paths(user_id, item_id, max_hops, max_paths)

        # Compute attention scores for each path
        path_explanations = []
        for path in paths:
            edges_with_scores = []
            path_score = 0.0

            for h, r, t in path:
                attention_score = self.get_attention_score(h, r, t)
                edges_with_scores.append(
                    {
                        "head": h,
                        "relation": r,
                        "tail": t,
                        "attention_score": attention_score,
                    }
                )
                path_score += attention_score

            path_explanations.append(
                {
                    "path": edges_with_scores,
                    "length": len(path),
                    "total_score": path_score,
                    "avg_score": path_score / len(path) if path else 0.0,
                }
            )

        # Sort by total score descending
        path_explanations.sort(key=lambda x: x["total_score"], reverse=True)

        return {
            "user_id": user_id,
            "item_id": item_id,
            "num_paths": len(paths),
            "paths": path_explanations,
        }

    def explain_correct_predictions(
        self,
        test_user_dict: Dict[int, List[int]],
        max_hops: int = 3,
        max_paths_per_pair: int = 10,
        top_k: int = 100,
        min_rank: int = 1,
    ) -> List[Dict]:
        """Explain all correct predictions in the test set.

        Args:
            test_user_dict: Test set dictionary mapping adjusted_user_id -> [item_ids]
            max_hops: Maximum number of hops to search
            max_paths_per_pair: Maximum paths per user-item pair
            top_k: Top-K predictions to consider
            min_rank: Minimum rank to consider as correct

        Returns:
            List of explanations for correct predictions
        """
        # Find correct predictions
        correct_predictions = self.predictor.find_correct_predictions(
            test_user_dict, top_k, min_rank
        )

        print(f"Found {len(correct_predictions)} correct predictions")

        # Explain each correct prediction
        explanations = []
        for i, (user_id, item_id, rank, score) in enumerate(correct_predictions):
            if (i + 1) % 10 == 0:
                print(f"Explaining prediction {i + 1}/{len(correct_predictions)}")

            explanation = self.explain_prediction(
                user_id, item_id, max_hops, max_paths_per_pair
            )
            explanation["rank"] = rank
            explanation["prediction_score"] = score
            explanations.append(explanation)

        return explanations

    def explain_top_predictions(
        self,
        user_id: int,
        max_hops: int = 3,
        max_paths_per_item: int = 10,
        top_k: int = 10,
    ) -> List[Dict]:
        """Explain top-k predictions for a user.

        Args:
            user_id: User ID (original ID, not adjusted)
            max_hops: Maximum number of hops to search
            max_paths_per_item: Maximum paths per item
            top_k: Number of top predictions to explain

        Returns:
            List of explanations for top predictions
        """
        # Get top-k predictions
        predicted_items, scores = self.predictor.predict_for_user(user_id, top_k)

        # Adjust user ID for entity offset
        adjusted_user_id = user_id + self.n_entities

        # Explain each prediction
        explanations = []
        for rank, (item_id, score) in enumerate(zip(predicted_items, scores), 1):
            explanation = self.explain_prediction(
                adjusted_user_id, item_id, max_hops, max_paths_per_item
            )
            explanation["rank"] = rank
            explanation["prediction_score"] = float(score)
            explanations.append(explanation)

        return explanations

    def get_path_statistics(
        self,
        test_user_dict: Dict[int, List[int]],
        max_hops: int = 3,
        sample_size: Optional[int] = None,
    ) -> Dict:
        """Get statistics about paths in correct predictions.

        Args:
            test_user_dict: Test set dictionary
            max_hops: Maximum number of hops
            sample_size: Number of pairs to sample (None for all)

        Returns:
            Dictionary with path statistics
        """
        # Find correct predictions
        correct_predictions = self.predictor.find_correct_predictions(
            test_user_dict, top_k=100, min_rank=1
        )

        if sample_size is not None and len(correct_predictions) > sample_size:
            import random

            correct_predictions = random.sample(correct_predictions, sample_size)

        # Collect statistics
        path_lengths = []
        path_counts = []
        has_path = 0
        no_path = 0

        for user_id, item_id, rank, score in correct_predictions:
            paths = self.path_finder.find_paths(user_id, item_id, max_hops)

            if paths:
                has_path += 1
                path_counts.append(len(paths))
                for path in paths:
                    path_lengths.append(len(path))
            else:
                no_path += 1

        import numpy as np

        return {
            "total_pairs": len(correct_predictions),
            "pairs_with_paths": has_path,
            "pairs_without_paths": no_path,
            "path_coverage": has_path / len(correct_predictions)
            if correct_predictions
            else 0,
            "avg_paths_per_pair": np.mean(path_counts) if path_counts else 0,
            "avg_path_length": np.mean(path_lengths) if path_lengths else 0,
            "min_path_length": min(path_lengths) if path_lengths else 0,
            "max_path_length": max(path_lengths) if path_lengths else 0,
            "total_paths": len(path_lengths),
        }

    def find_most_important_edges(
        self, explanations: List[Dict], top_k: int = 20
    ) -> List[Tuple[int, int, int, float, int]]:
        """Find the most important edges across all explanations.

        Args:
            explanations: List of explanations from explain_correct_predictions
            top_k: Number of top edges to return

        Returns:
            List of (head, relation, tail, avg_score, frequency) tuples
        """
        edge_scores = {}
        edge_counts = {}

        for explanation in explanations:
            for path_info in explanation["paths"]:
                for edge in path_info["path"]:
                    h = edge["head"]
                    r = edge["relation"]
                    t = edge["tail"]
                    score = edge["attention_score"]

                    key = (h, r, t)
                    if key not in edge_scores:
                        edge_scores[key] = 0.0
                        edge_counts[key] = 0

                    edge_scores[key] += score
                    edge_counts[key] += 1

        # Calculate average scores and create result list
        important_edges = []
        for (h, r, t), total_score in edge_scores.items():
            count = edge_counts[(h, r, t)]
            avg_score = total_score / count
            important_edges.append((h, r, t, avg_score, count))

        # Sort by frequency first, then by average score
        important_edges.sort(key=lambda x: (x[4], x[3]), reverse=True)

        return important_edges[:top_k]
