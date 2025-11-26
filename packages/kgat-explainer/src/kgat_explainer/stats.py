"""Statistics and evaluation functions for KGAT Explainer."""

import logging

import torch
from kgat.config import KGATConfig
from kgat.core.kgat import KGAT
from kgat.data.dataloader import DataLoader

from kgat_explainer.config import KGATExplainerConfig
from kgat_explainer.explainer import KGATExplainer
from kgat_explainer.model_loader import KGATModelLoader


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


def compute_statistics(config: KGATExplainerConfig):
    """Compute path statistics for a trained KGAT model.

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

    # Initialize explainer (no attention precomputation needed for stats)
    logger.info("Initializing KGAT Explainer...")
    explainer = KGATExplainer(
        model=model,
        data_loader=data_loader,
        kg_dict=data_loader.train_kg_dict,
        kg_dict_by_relation=kg_dict_by_relation,
        device=device,
        precompute_attention=False,
    )

    # Get path statistics
    sample_size = config.sample_size if config.sample_size else 100
    logger.info(f"Computing path statistics (sample size: {sample_size})...")
    stats = explainer.get_path_statistics(
        test_user_dict=data_loader.test_user_dict,
        max_hops=config.max_hops,
        sample_size=sample_size,
    )

    # Evaluate metrics
    logger.info("Evaluating model metrics...")
    k_list = eval(config.Ks)
    metrics = explainer.predictor.evaluate_metrics(
        test_user_dict=data_loader.test_user_dict,
        k_list=k_list,
    )

    # Print results
    logger.info("\n=== Path Statistics ===")
    logger.info(f"Total pairs sampled: {stats['total_pairs']}")
    logger.info(f"Pairs with paths: {stats['pairs_with_paths']}")
    logger.info(f"Pairs without paths: {stats['pairs_without_paths']}")
    logger.info(f"Path coverage: {stats['path_coverage']:.2%}")
    logger.info(f"Average paths per pair: {stats['avg_paths_per_pair']:.2f}")
    logger.info(f"Average path length: {stats['avg_path_length']:.2f}")
    logger.info(f"Min path length: {stats['min_path_length']}")
    logger.info(f"Max path length: {stats['max_path_length']}")
    logger.info(f"Total paths found: {stats['total_paths']}")

    logger.info("\n=== Model Performance ===")
    for k, metric_dict in metrics.items():
        logger.info(
            f"K={k:3d}: Precision={metric_dict['precision']:.4f}, "
            f"Recall={metric_dict['recall']:.4f}, "
            f"NDCG={metric_dict['ndcg']:.4f}"
        )


def evaluate_model(config: KGATExplainerConfig):
    """Evaluate a trained KGAT model on test set.

    Args:
        config: Explainer configuration
    """
    logger = setup_logging(config.verbose)

    # Load model and data
    model, data_loader, device = load_model_and_data(config)

    # Prepare KG data structures
    kg_dict_by_relation = {}
    for relation, ht_list in data_loader.train_relation_dict.items():
        kg_dict_by_relation[relation] = ht_list

    # Initialize explainer (only for predictor)
    logger.info("Initializing predictor...")
    explainer = KGATExplainer(
        model=model,
        data_loader=data_loader,
        kg_dict=data_loader.train_kg_dict,
        kg_dict_by_relation=kg_dict_by_relation,
        device=device,
        precompute_attention=False,
    )

    # Evaluate
    logger.info("Evaluating model...")
    k_list = eval(config.Ks)
    metrics = explainer.predictor.evaluate_metrics(
        test_user_dict=data_loader.test_user_dict,
        k_list=k_list,
    )

    # Print results
    logger.info("\n=== Evaluation Results ===")
    logger.info(f"Dataset: {config.data_name}")
    logger.info(f"Test users: {len(data_loader.test_user_dict)}")
    logger.info(f"Test interactions: {data_loader.n_cf_test}")
    logger.info("")
    logger.info("Metrics:")
    for k, metric_dict in metrics.items():
        logger.info(
            f"  K={k:3d}: Precision={metric_dict['precision']:.4f}, "
            f"Recall={metric_dict['recall']:.4f}, "
            f"NDCG={metric_dict['ndcg']:.4f}"
        )
