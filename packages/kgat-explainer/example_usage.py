"""Example usage script for KGAT Explainer."""

import logging
import os
from pathlib import Path

import torch
from kgat.config import KGATConfig
from kgat.core.kgat import KGAT
from kgat.data.dataloader import DataLoader
from kgat_explainer import KGATExplainer, KGATModelLoader


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def load_trained_model(
    config: KGATConfig, data_loader: DataLoader, device: torch.device
):
    """Load a trained KGAT model.

    Args:
        config: KGAT configuration
        data_loader: Data loader with KG and CF data
        device: Device to load model on

    Returns:
        Loaded KGAT model
    """
    # Find the model file
    model_dir = Path("packages/kgat/trained_model/KGAT") / config.data_name

    # Look for model files
    model_files = list(model_dir.rglob("*.pth"))

    if not model_files:
        raise FileNotFoundError(f"No trained model found in {model_dir}")

    # Use the first model file found
    model_path = str(model_files[0])
    print(f"Loading model from: {model_path}")

    # Get pretrained embeddings if needed
    user_pre_embed = None
    item_pre_embed = None
    if config.use_pretrain == 1 and hasattr(data_loader, "user_pre_embed"):
        user_pre_embed = torch.FloatTensor(data_loader.user_pre_embed)
        item_pre_embed = torch.FloatTensor(data_loader.item_pre_embed)

    # Initialize model loader
    loader = KGATModelLoader(model_path, device)

    # Load the model
    model = loader.load_model(
        model_class=KGAT,
        config=config,
        n_users=data_loader.n_users,
        n_entities=data_loader.n_entities,
        n_relations=data_loader.n_relations,
        A_in=data_loader.A_in.to(device),
        user_pre_embed=user_pre_embed,
        item_pre_embed=item_pre_embed,
    )

    # Get model info
    model_info = loader.get_model_info()
    print(f"Model info: {model_info}")

    return model


def main():
    """Main function demonstrating KGAT Explainer usage."""
    logger = setup_logging()

    # Configuration
    config = KGATConfig(
        data_name="yelp",
        data_dir="datasets/",
        use_pretrain=1,
        embed_dim=64,
        relation_dim=64,
        aggregation_type="bi-interaction",
        conv_dim_list="[64,32,16]",
        mess_dropout="[0.1,0.1,0.1]",
    )

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading data...")
    data_loader = DataLoader(config, logger)

    # Load trained model
    logger.info("Loading trained model...")
    model = load_trained_model(config, data_loader, device)

    # Prepare KG dictionary by relation for efficient attention computation
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
        precompute_attention=True,  # Precompute all attention scores
    )

    # Example 1: Get path statistics
    logger.info("\n=== Example 1: Path Statistics ===")
    stats = explainer.get_path_statistics(
        test_user_dict=data_loader.test_user_dict,
        max_hops=3,
        sample_size=100,  # Sample 100 pairs for statistics
    )
    logger.info(f"Path statistics: {stats}")

    # Example 2: Explain correct predictions
    logger.info("\n=== Example 2: Explain Correct Predictions ===")
    explanations = explainer.explain_correct_predictions(
        test_user_dict=data_loader.test_user_dict,
        max_hops=3,
        max_paths_per_pair=10,
        top_k=20,
        min_rank=1,
    )

    # Show first few explanations
    for i, explanation in enumerate(explanations[:3]):
        logger.info(f"\nExplanation {i + 1}:")
        logger.info(f"  User ID: {explanation['user_id']}")
        logger.info(f"  Item ID: {explanation['item_id']}")
        logger.info(f"  Rank: {explanation['rank']}")
        logger.info(f"  Prediction Score: {explanation['prediction_score']:.4f}")
        logger.info(f"  Number of paths: {explanation['num_paths']}")

        if explanation["paths"]:
            best_path = explanation["paths"][0]
            logger.info(f"  Best path (score: {best_path['total_score']:.4f}):")
            for j, edge in enumerate(best_path["path"]):
                logger.info(
                    f"    Edge {j + 1}: {edge['head']} --[{edge['relation']}]--> "
                    f"{edge['tail']} (attention: {edge['attention_score']:.4f})"
                )

    # Example 3: Find most important edges
    logger.info("\n=== Example 3: Most Important Edges ===")
    important_edges = explainer.find_most_important_edges(explanations, top_k=10)

    for i, (h, r, t, avg_score, freq) in enumerate(important_edges, 1):
        logger.info(
            f"  {i}. Edge ({h}, {r}, {t}): avg_score={avg_score:.4f}, frequency={freq}"
        )

    # Example 4: Explain top predictions for a specific user
    logger.info("\n=== Example 4: Explain Top Predictions for a User ===")

    # Get a test user
    test_users = list(data_loader.test_user_dict.keys())
    if test_users:
        adjusted_user_id = test_users[0]
        original_user_id = adjusted_user_id - data_loader.n_entities

        logger.info(f"Explaining predictions for user {original_user_id}")

        top_explanations = explainer.explain_top_predictions(
            user_id=original_user_id,
            max_hops=3,
            max_paths_per_item=5,
            top_k=5,
        )

        for exp in top_explanations:
            logger.info(
                f"\n  Rank {exp['rank']}: Item {exp['item_id']} "
                f"(score: {exp['prediction_score']:.4f}, "
                f"paths: {exp['num_paths']})"
            )

            if exp["paths"]:
                best_path = exp["paths"][0]
                logger.info(f"    Best path score: {best_path['total_score']:.4f}")

    # Example 5: Evaluate model performance
    logger.info("\n=== Example 5: Model Evaluation Metrics ===")
    metrics = explainer.predictor.evaluate_metrics(
        test_user_dict=data_loader.test_user_dict,
        k_list=[20, 40, 60, 80, 100],
    )

    for k, metric_dict in metrics.items():
        logger.info(
            f"  K={k}: Precision={metric_dict['precision']:.4f}, "
            f"Recall={metric_dict['recall']:.4f}, "
            f"NDCG={metric_dict['ndcg']:.4f}"
        )

    logger.info("\n=== Done ===")


if __name__ == "__main__":
    main()
