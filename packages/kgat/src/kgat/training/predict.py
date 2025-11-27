import logging

import numpy as np
import torch

from kgat.config import KGATConfig
from kgat.core.kgat import KGAT
from kgat.data.dataloader import DataLoader
from kgat.training.evaluate import evaluate
from kgat.utils.model_helper import load_model


def predict(config: KGATConfig):
    """
    Prediction mode: Evaluate a trained model on the TEST set.

    This mode is used for:
    - Final performance evaluation on the test set
    - Generating prediction scores for all user-item pairs

    Note: This evaluates on TEST set (use_validation=False)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = "packages/kgat/trained_model/KGAT/{}/embed-dim{}_relation-dim{}_{}_{}_{}_lr{}_pretrain{}/".format(
        config.data_name,
        config.embed_dim,
        config.relation_dim,
        config.laplacian_type,
        config.aggregation_type,
        "-".join([str(i) for i in eval(config.conv_dim_list)]),
        config.lr,
        config.use_pretrain,
    )

    logger = logging.getLogger(__name__)

    data = DataLoader(config, logger)

    model = KGAT(config, data.n_users, data.n_entities, data.n_relations)
    model = load_model(model, config.pretrain_model_path)
    model.to(device)

    Ks = eval(config.Ks)
    k_min = min(Ks)
    k_max = max(Ks)

    # Evaluate on TEST set (use_validation=False)
    cf_scores, metrics_dict = evaluate(model, data, Ks, device, use_validation=False)
    np.save(save_dir + "cf_scores.npy", cf_scores)
    print(
        "TEST Set Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]".format(
            metrics_dict[k_min]["precision"],
            metrics_dict[k_max]["precision"],
            metrics_dict[k_min]["recall"],
            metrics_dict[k_max]["recall"],
            metrics_dict[k_min]["ndcg"],
            metrics_dict[k_max]["ndcg"],
        )
    )
