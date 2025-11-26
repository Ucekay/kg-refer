"""Model loader for trained KGAT models."""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class KGATModelLoader:
    """Loads trained KGAT models and associated data."""

    def __init__(
        self,
        model_path: str,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize the model loader.

        Args:
            model_path: Path to the trained model file (.pth)
            device: Device to load the model on
        """
        self.model_path = model_path
        self.device = device

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

    def load_model(
        self,
        model_class: type,
        config,
        n_users: int,
        n_entities: int,
        n_relations: int,
        A_in: Optional[torch.Tensor] = None,
        user_pre_embed: Optional[torch.Tensor] = None,
        item_pre_embed: Optional[torch.Tensor] = None,
    ) -> nn.Module:
        """Load a trained KGAT model.

        Args:
            model_class: KGAT model class
            config: KGAT configuration object
            n_users: Number of users
            n_entities: Number of entities
            n_relations: Number of relations
            A_in: Adjacency matrix (optional)
            user_pre_embed: User pre-trained embeddings (optional)
            item_pre_embed: Item pre-trained embeddings (optional)

        Returns:
            Loaded KGAT model
        """
        # Initialize model
        model = model_class(
            config=config,
            n_users=n_users,
            n_entities=n_entities,
            n_relations=n_relations,
            A_in=A_in,
            user_pre_embed=user_pre_embed,
            item_pre_embed=item_pre_embed,
        )

        # Load state dict
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()

        return model

    @staticmethod
    def load_checkpoint(
        checkpoint_path: str, device: torch.device = torch.device("cpu")
    ) -> Dict:
        """Load a checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load on

        Returns:
            Checkpoint dictionary
        """
        return torch.load(checkpoint_path, map_location=device)

    @staticmethod
    def save_model(
        model: nn.Module,
        save_path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        metrics: Optional[Dict] = None,
    ):
        """Save a model checkpoint.

        Args:
            model: Model to save
            save_path: Path to save the model
            optimizer: Optimizer state (optional)
            epoch: Current epoch (optional)
            metrics: Training metrics (optional)
        """
        checkpoint = {"model_state_dict": model.state_dict()}

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if epoch is not None:
            checkpoint["epoch"] = epoch

        if metrics is not None:
            checkpoint["metrics"] = metrics

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        torch.save(checkpoint, save_path)

    def get_model_info(self) -> Dict:
        """Get information about the saved model.

        Returns:
            Dictionary with model information
        """
        checkpoint = torch.load(self.model_path, map_location="cpu")

        info = {
            "file_path": self.model_path,
            "file_size_mb": os.path.getsize(self.model_path) / (1024 * 1024),
        }

        if isinstance(checkpoint, dict):
            if "epoch" in checkpoint:
                info["epoch"] = checkpoint["epoch"]
            if "metrics" in checkpoint:
                info["metrics"] = checkpoint["metrics"]
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Count parameters
        total_params = sum(p.numel() for p in state_dict.values())
        info["total_parameters"] = total_params

        return info
