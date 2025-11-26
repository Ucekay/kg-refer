"""Configuration for KGAT Explainer using cyclopts."""

from dataclasses import dataclass
from typing import Literal, Optional

from cyclopts import Parameter


@Parameter(name="*")
@dataclass
class KGATExplainerConfig:
    """Configuration for KGAT Explainer."""

    # Model and data paths
    model_path: str = "packages/kgat/trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain0/model_epoch300.pth"
    "Path to trained KGAT model (.pth file)"

    data_name: str = "yelp"
    "Dataset name"

    data_dir: str = "datasets/"
    "Directory containing dataset"

    output_dir: str = "output/explanations/"
    "Directory to save explanations"

    # KGAT model configuration (must match the trained model)
    use_pretrain: int = 0
    "0: No pretrain, 1: Pretrain with learned embeddings, 2: Pretrain with stored model"

    pretrain_embedding_dir: str = "packages/kgat/trained/pretrain/"
    "Path of learned embeddings"

    embed_dim: int = 64
    "User-entity embedding size"

    relation_dim: int = 64
    "Relation embedding size"

    laplacian_type: Literal["random-walk", "symmetric"] = "random-walk"
    "Specify the type of the adjacency (laplacian) matrix"

    aggregation_type: Literal["gcn", "graphsage", "bi-interaction"] = "bi-interaction"
    "Specify the type of the aggregation layer"

    conv_dim_list: str = "[64,32,16]"
    "Output size of every aggregation layer"

    mess_dropout: str = "[0.1,0.1,0.1]"
    "Dropout probability of every aggregation layer"

    kg_l2loss_lambda: float = 1e-5
    "Lambda for kg l2 loss"

    cf_l2loss_lambda: float = 1e-5
    "Lambda for cf l2 loss"

    # Path finding parameters
    max_hops: int = 3
    "Maximum number of hops for path finding"

    max_paths_per_pair: int = 10
    "Maximum number of paths per user-item pair"

    # Prediction parameters
    top_k: int = 100
    "Top-K predictions to consider"

    min_rank: int = 1
    "Minimum rank to consider as correct prediction"

    sample_size: Optional[int] = None
    "Number of correct predictions to explain (None for all)"

    # Evaluation parameters
    Ks: str = "[20, 40, 60, 80, 100]"
    "Calculate metrics@K when evaluating"

    # Computation settings
    precompute_attention: bool = True
    "Precompute all attention scores (faster but uses more memory)"

    use_cuda: bool = False
    "Use CUDA if available"

    seed: int = 2019
    "Random seed"

    # Modes
    explain_correct: bool = False
    "Explain correct predictions mode"

    explain_user: Optional[int] = None
    "Explain top predictions for a specific user (user ID)"

    statistics: bool = False
    "Compute path statistics only"

    evaluate: bool = False
    "Evaluate model performance only"

    # Output settings
    save_explanations: bool = True
    "Save explanations to JSON files"

    verbose: bool = False
    "Verbose logging"
