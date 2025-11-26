"""Predictor module for KGAT model evaluation and correct prediction identification."""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from kgat.training.evaluate import evaluate


class KGATPredictor:
    """Handles prediction and evaluation of KGAT model."""

    def __init__(
        self,
        model: nn.Module,
        data_loader,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize the predictor.

        Args:
            model: Trained KGAT model
            data_loader: KGAT DataLoader with train/val/test data
            device: Device to run computations on
        """
        self.model = model
        self.data_loader = data_loader
        self.n_users = data_loader.n_users
        self.n_items = data_loader.n_items
        self.n_entities = data_loader.n_entities
        self.device = device
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict_for_user(
        self, user_id: int, top_k: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict top-k items for a user.

        Args:
            user_id: User ID (original ID, will be adjusted for entity offset)
            top_k: Number of top items to return

        Returns:
            Tuple of (item_ids, scores) sorted by score descending
        """
        # Adjust user ID for entity offset
        adjusted_user_id = user_id + self.n_entities

        # Get all item IDs
        item_ids = torch.arange(self.n_items).to(self.device)

        # Create user tensor
        user_tensor = torch.LongTensor([adjusted_user_id]).to(self.device)

        # Predict scores
        scores = self.model(user_tensor, item_ids, mode="predict")
        scores = scores.squeeze().cpu().numpy()

        # Get top-k items
        top_k_idx = np.argsort(scores)[::-1][:top_k]
        top_k_items = item_ids.cpu().numpy()[top_k_idx]
        top_k_scores = scores[top_k_idx]

        return top_k_items, top_k_scores

    def evaluate_metrics(
        self,
        test_user_dict: Dict[int, List[int]],
        k_list: List[int] = [20, 40, 60, 80, 100],
        use_validation: bool = False,
    ) -> Dict[int, Dict[str, float]]:
        """Evaluate the model with Precision, Recall, NDCG at K.

        This method reuses KGAT's evaluate function for consistency.

        Args:
            test_user_dict: Dictionary mapping adjusted_user_id -> [item_ids]
                           (not used - kept for API compatibility)
            k_list: List of K values for evaluation
            use_validation: If True, evaluate on validation set; if False, on test set

        Returns:
            Dictionary mapping K -> metrics dictionary
        """
        # Use KGAT's evaluate function
        _, metrics_dict = evaluate(
            self.model, self.data_loader, k_list, self.device, use_validation
        )
        return metrics_dict

    def find_correct_predictions(
        self,
        test_user_dict: Dict[int, List[int]],
        top_k: int = 100,
        min_rank: int = 1,
    ) -> List[Tuple[int, int, int, float]]:
        """Find user-item pairs where the model correctly predicted the item.

        Args:
            test_user_dict: Dictionary mapping adjusted_user_id -> [item_ids]
            top_k: Number of top predictions to consider
            min_rank: Minimum rank to consider as correct (1-indexed)

        Returns:
            List of (user_id, item_id, rank, score) for correct predictions
        """
        correct_predictions = []

        for adjusted_user_id, ground_truth_items in test_user_dict.items():
            # Original user ID
            user_id = adjusted_user_id - self.n_entities

            # Get predictions
            predicted_items, scores = self.predict_for_user(user_id, top_k=top_k)

            # Check which ground truth items are in top-k
            for gt_item in ground_truth_items:
                if gt_item in predicted_items:
                    rank = np.where(predicted_items == gt_item)[0][0] + 1
                    if rank >= min_rank:
                        score = scores[rank - 1]
                        correct_predictions.append(
                            (adjusted_user_id, gt_item, rank, score)
                        )

        return correct_predictions

    def find_correct_predictions_with_threshold(
        self,
        test_user_dict: Dict[int, List[int]],
        score_threshold: float = 0.0,
        top_k: int = 100,
    ) -> List[Tuple[int, int, int, float]]:
        """Find correct predictions above a score threshold.

        Args:
            test_user_dict: Dictionary mapping adjusted_user_id -> [item_ids]
            score_threshold: Minimum score threshold for predictions
            top_k: Number of top predictions to consider

        Returns:
            List of (user_id, item_id, rank, score) for correct predictions
        """
        correct_predictions = []

        for adjusted_user_id, ground_truth_items in test_user_dict.items():
            # Original user ID
            user_id = adjusted_user_id - self.n_entities

            # Get predictions
            predicted_items, scores = self.predict_for_user(user_id, top_k=top_k)

            # Check which ground truth items are in top-k with sufficient score
            for gt_item in ground_truth_items:
                if gt_item in predicted_items:
                    rank = np.where(predicted_items == gt_item)[0][0] + 1
                    score = scores[rank - 1]
                    if score >= score_threshold:
                        correct_predictions.append(
                            (adjusted_user_id, gt_item, rank, score)
                        )

        return correct_predictions

    @torch.no_grad()
    def predict_batch(
        self, user_ids: np.ndarray, batch_size: int = 100
    ) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Predict for multiple users in batches.

        Args:
            user_ids: Array of user IDs (original IDs)
            batch_size: Batch size for processing

        Returns:
            Dictionary mapping user_id -> (predicted_items, scores)
        """
        predictions = {}

        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i : i + batch_size]

            for user_id in batch_users:
                items, scores = self.predict_for_user(user_id)
                predictions[user_id] = (items, scores)

        return predictions

    def get_all_predictions(
        self, use_validation: bool = False
    ) -> Tuple[np.ndarray, Dict[int, Dict[str, float]]]:
        """Get predictions for all users using KGAT's evaluate function.

        Args:
            use_validation: If True, evaluate on validation set; if False, on test set

        Returns:
            Tuple of (cf_scores, metrics_dict)
            - cf_scores: Array of shape (n_users, n_items) with prediction scores
            - metrics_dict: Dictionary of evaluation metrics
        """
        Ks = [20, 40, 60, 80, 100]
        cf_scores, metrics_dict = evaluate(
            self.model, self.data_loader, Ks, self.device, use_validation
        )
        return cf_scores, metrics_dict
