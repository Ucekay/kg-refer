"""Explanation functions for KGAT Explainer."""

import json
import logging
from pathlib import Path

import numpy as np
import torch
from kgat.config import KGATConfig
from kgat.core.kgat import KGAT
from kgat.data.dataloader import DataLoader

from kgat_explainer.config import KGATExplainerConfig
from kgat_explainer.explainer import KGATExplainer
from kgat_explainer.model_loader import KGATModelLoader


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy and PyTorch data types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (torch.Tensor,)):
            return obj.cpu().numpy().tolist()
        return super().default(obj)


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_model_and_data(config: KGATExplainerConfig):
    """Load model and data.

    Args:
        config: Explainer configuration

    Returns:
        Tuple of (model, data_loader, device)
    """
    logger = logging.getLogger(__name__)

    # Check if model file exists
    if not Path(config.model_path).exists():
        raise FileNotFoundError(f"Model file not found: {config.model_path}")

    # Set device
    device = torch.device(
        "cuda" if config.use_cuda and torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create KGAT configuration
    kgat_config = KGATConfig(
        data_name=config.data_name,
        data_dir=config.data_dir,
        use_pretrain=config.use_pretrain,
        pretrain_embedding_dir=config.pretrain_embedding_dir,
        embed_dim=config.embed_dim,
        relation_dim=config.relation_dim,
        laplacian_type=config.laplacian_type,
        aggregation_type=config.aggregation_type,
        conv_dim_list=config.conv_dim_list,
        mess_dropout=config.mess_dropout,
        kg_l2loss_lambda=config.kg_l2loss_lambda,
        cf_l2loss_lambda=config.cf_l2loss_lambda,
    )

    # Load data
    logger.info("Loading data...")
    data_loader = DataLoader(kgat_config, logger)

    # Get pretrained embeddings if needed
    user_pre_embed = None
    item_pre_embed = None
    if config.use_pretrain == 1 and hasattr(data_loader, "user_pre_embed"):
        user_pre_embed = torch.FloatTensor(data_loader.user_pre_embed)
        item_pre_embed = torch.FloatTensor(data_loader.item_pre_embed)

    # Load model
    logger.info(f"Loading model from {config.model_path}...")
    model_loader = KGATModelLoader(config.model_path, device)
    model = model_loader.load_model(
        model_class=KGAT,
        config=kgat_config,
        n_users=data_loader.n_users,
        n_entities=data_loader.n_entities,
        n_relations=data_loader.n_relations,
        A_in=data_loader.A_in.to(device),
        user_pre_embed=user_pre_embed,
        item_pre_embed=item_pre_embed,
    )

    return model, data_loader, device


def explain_correct_predictions(config: KGATExplainerConfig):
    """Explain correct predictions from a trained KGAT model.

    Args:
        config: Explainer configuration
    """
    logger = setup_logging(config.verbose)

    # Load model and data
    model, data_loader, device = load_model_and_data(config)

    # Prepare KG data structures
    logger.info("Preparing KG data structures...")
    kg_dict_by_relation = {}
    for relation, ht_list in data_loader.train_relation_dict.items():
        kg_dict_by_relation[relation] = ht_list

    # Initialize explainer
    logger.info("Initializing KGAT Explainer...")
    explainer = KGATExplainer(
        model=model,
        data_loader=data_loader,
        kg_dict=data_loader.train_kg_dict,
        kg_dict_by_relation=kg_dict_by_relation,
        device=device,
        precompute_attention=config.precompute_attention,
    )

    # Get path statistics
    logger.info("Computing path statistics...")
    stats = explainer.get_path_statistics(
        test_user_dict=data_loader.test_user_dict,
        max_hops=config.max_hops,
        sample_size=100,
    )
    logger.info(f"Path statistics: {json.dumps(stats, indent=2)}")

    # Explain correct predictions
    logger.info("Explaining correct predictions...")
    explanations = explainer.explain_correct_predictions(
        test_user_dict=data_loader.test_user_dict,
        max_hops=config.max_hops,
        max_paths_per_pair=config.max_paths_per_pair,
        top_k=config.top_k,
        min_rank=config.min_rank,
    )

    # Limit to sample size if specified
    if config.sample_size is not None and len(explanations) > config.sample_size:
        explanations = explanations[: config.sample_size]

    logger.info(f"Generated {len(explanations)} explanations")

    # Find most important edges
    logger.info("Finding most important edges...")
    important_edges = explainer.find_most_important_edges(explanations, top_k=20)

    # Save results if requested
    if config.save_explanations:
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save explanations
        explanations_file = output_path / f"{config.data_name}_explanations.json"
        with open(explanations_file, "w") as f:
            json.dump(explanations, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Saved explanations to {explanations_file}")

        # Save statistics
        stats_file = output_path / f"{config.data_name}_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(
                {
                    "path_statistics": stats,
                    "num_explanations": len(explanations),
                    "important_edges": [
                        {
                            "head": int(h),
                            "relation": int(r),
                            "tail": int(t),
                            "avg_score": float(score),
                            "frequency": int(freq),
                        }
                        for h, r, t, score, freq in important_edges
                    ],
                },
                f,
                indent=2,
                cls=NumpyEncoder,
            )
        logger.info(f"Saved statistics to {stats_file}")

    # Print summary
    logger.info("\n=== Summary ===")
    logger.info(f"Total explanations: {len(explanations)}")
    logger.info(f"Path coverage: {stats['path_coverage']:.2%}")
    logger.info(f"Average paths per pair: {stats['avg_paths_per_pair']:.2f}")
    logger.info(f"Average path length: {stats['avg_path_length']:.2f}")
    logger.info("\nTop 5 most important edges:")
    for i, (h, r, t, score, freq) in enumerate(important_edges[:5], 1):
        logger.info(f"  {i}. ({h}, {r}, {t}): score={score:.4f}, freq={freq}")


def explain_user_predictions(config: KGATExplainerConfig):
    """Explain top predictions for a specific user.

    Args:
        config: Explainer configuration
    """
    logger = setup_logging(config.verbose)

    if config.explain_user is None:
        logger.error("User ID not specified")
        return

    # Load model and data
    model, data_loader, device = load_model_and_data(config)

    # Prepare KG data structures
    logger.info("Preparing KG data structures...")
    kg_dict_by_relation = {}
    for relation, ht_list in data_loader.train_relation_dict.items():
        kg_dict_by_relation[relation] = ht_list

    # Initialize explainer
    logger.info("Initializing KGAT Explainer...")
    explainer = KGATExplainer(
        model=model,
        data_loader=data_loader,
        kg_dict=data_loader.train_kg_dict,
        kg_dict_by_relation=kg_dict_by_relation,
        device=device,
        precompute_attention=config.precompute_attention,
    )

    # Explain top predictions
    logger.info(f"Explaining predictions for user {config.explain_user}...")
    top_explanations = explainer.explain_top_predictions(
        user_id=config.explain_user,
        max_hops=config.max_hops,
        max_paths_per_item=config.max_paths_per_pair,
        top_k=config.top_k,
    )

    # Print results
    logger.info(
        f"\n=== Top {config.top_k} Predictions for User {config.explain_user} ==="
    )
    for exp in top_explanations:
        logger.info(
            f"\nRank {exp['rank']}: Item {exp['item_id']} "
            f"(score: {exp['prediction_score']:.4f}, paths: {exp['num_paths']})"
        )

        if exp["paths"]:
            best_path = exp["paths"][0]
            logger.info(f"  Best path score: {best_path['total_score']:.4f}")
            for j, edge in enumerate(best_path["path"]):
                logger.info(
                    f"    Edge {j + 1}: {edge['head']} --[{edge['relation']}]--> "
                    f"{edge['tail']} (attention: {edge['attention_score']:.4f})"
                )

    # Save results if requested
    if config.save_explanations:
        output_path = Path(config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        explanations_file = (
            output_path
            / f"{config.data_name}_user{config.explain_user}_explanations.json"
        )
        with open(explanations_file, "w") as f:
            json.dump(top_explanations, f, indent=2, cls=NumpyEncoder)
        logger.info(f"\nSaved explanations to {explanations_file}")
