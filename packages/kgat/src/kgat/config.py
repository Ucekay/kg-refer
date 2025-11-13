from dataclasses import dataclass
from typing import Literal

from cyclopts import Parameter


@Parameter(name="*")
@dataclass
class KGATConfig:
    seed: int = 2019
    "Random seed"
    data_name: str = "yelp"
    "Cheese a dataset"
    data_dir: str = "datasets/"
    "Directory to load data"

    use_pretrain: int = 1
    "0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stores model"
    pretrain_embedding_dir: str = "packages/kgat/trained/pretrain/"
    "Path of learned embeddings"
    pretrain_model_path: str = "packages/kgat/trained/model.pth"
    "Path of stored model"

    cf_batch_size: int = 1024
    "Batch size for cf"
    kg_batch_size: int = 2048
    "Batch size for kg"
    test_batch_size: int = 10000
    "Batch size for testing (the user number to test every batch"

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

    lr: float = 0.0001
    "Learning rate"
    n_epoch: int = 1000
    "Number of epochs"
    stopping_steps: int = 10
    "Early stopping steps"

    cf_print_every: int = 1
    "Iter interval for printing cf loss"
    kg_print_every: int = 1
    "Iter interval for printing kg loss"
    evaluate_every: int = 10
    "Epoch interval for evaluating the cf"

    Ks: str = "[20, 40, 60, 80, 100]"
    "Calculate metrics@K when evaluating"

    train: bool = False
    "Training mode"
    predict: bool = False
    "Prediction mode"
