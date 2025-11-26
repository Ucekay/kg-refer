"""KGAT Explainer package for explaining KGAT model predictions through path analysis."""

from kgat_explainer.attention import AttentionCalculator
from kgat_explainer.config import KGATExplainerConfig
from kgat_explainer.explainer import KGATExplainer
from kgat_explainer.model_loader import KGATModelLoader
from kgat_explainer.path_finder import PathFinder
from kgat_explainer.predictor import KGATPredictor

__all__ = [
    "AttentionCalculator",
    "KGATExplainer",
    "KGATExplainerConfig",
    "KGATModelLoader",
    "PathFinder",
    "KGATPredictor",
]

__version__ = "0.1.0"
