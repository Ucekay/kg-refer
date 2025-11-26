"""Simple test script for KGAT Explainer functionality."""

import logging
import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kgat.config import KGATConfig
from kgat.core.kgat import KGAT
from kgat.data.dataloader import DataLoader
from kgat_explainer import (
    AttentionCalculator,
    KGATExplainer,
    KGATModelLoader,
    KGATPredictor,
    PathFinder,
)


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


def test_attention_calculator():
    """Test AttentionCalculator functionality."""
    print("\n=== Testing AttentionCalculator ===")

    # Create dummy embeddings
    entity_user_embed = torch.nn.Embedding(100, 64)
    relation_embed = torch.nn.Embedding(10, 64)
    trans_M = torch.nn.Parameter(torch.randn(10, 64, 64))

    calculator = AttentionCalculator(
        entity_user_embed, relation_embed, trans_M, torch.device("cpu")
    )

    # Test single edge attention
    score = calculator.calculate_edge_attention(0, 1, 0)
    print(f"✓ Single edge attention score: {score:.4f}")

    # Test batch attention
    h_list = torch.LongTensor([0, 1, 2])
    t_list = torch.LongTensor([3, 4, 5])
    scores = calculator.calculate_batch_attention(h_list, t_list, 0)
    print(f"✓ Batch attention scores: {scores.shape}")

    # Test precompute
    kg_dict_by_relation = {0: [(0, 1), (1, 2), (2, 3)], 1: [(0, 2), (1, 3)]}
    attention_scores = calculator.precompute_attention_scores_by_relation(
        kg_dict_by_relation
    )
    print(f"✓ Precomputed {len(attention_scores)} attention scores")


def test_path_finder():
    """Test PathFinder functionality."""
    print("\n=== Testing PathFinder ===")

    # Create a simple KG
    kg_dict = {
        0: [(1, 0), (2, 1)],  # 0 -> 1, 0 -> 2
        1: [(3, 0), (4, 1)],  # 1 -> 3, 1 -> 4
        2: [(3, 0), (5, 1)],  # 2 -> 3, 2 -> 5
        3: [(4, 0)],  # 3 -> 4
    }

    path_finder = PathFinder(kg_dict, n_entities=10)

    # Test finding paths
    paths = path_finder.find_paths(0, 4, max_hops=3)
    print(f"✓ Found {len(paths)} paths from 0 to 4")
    for i, path in enumerate(paths[:3]):
        print(f"  Path {i + 1}: {[(h, r, t) for h, r, t in path]}")

    # Test shortest paths
    shortest = path_finder.find_shortest_paths(0, 4, max_hops=3)
    print(f"✓ Found {len(shortest)} shortest paths")

    # Test path counting
    count = path_finder.count_paths(0, 4, max_hops=3)
    print(f"✓ Path count: {count}")


def test_predictor():
    """Test KGATPredictor functionality."""
    print("\n=== Testing KGATPredictor ===")

    # Create dummy model and mock data_loader
    config = KGATConfig(
        embed_dim=64,
        relation_dim=64,
        aggregation_type="bi-interaction",
        conv_dim_list="[64,32,16]",
        mess_dropout="[0.1,0.1,0.1]",
    )

    model = KGAT(
        config=config,
        n_users=100,
        n_entities=200,
        n_relations=10,
    )
    model.eval()

    # Create a mock data_loader
    class MockDataLoader:
        def __init__(self):
            self.n_users = 100
            self.n_items = 150
            self.n_entities = 200
            self.test_batch_size = 10
            self.train_user_dict = {}
            self.test_user_dict = {
                200: [0, 5, 10],  # User 0 (adjusted)
                201: [1, 2],  # User 1 (adjusted)
            }
            self.val_user_dict = {}

    data_loader = MockDataLoader()

    predictor = KGATPredictor(
        model=model,
        data_loader=data_loader,
        device=torch.device("cpu"),
    )

    # Test prediction for single user
    items, scores = predictor.predict_for_user(0, top_k=10)
    print(f"✓ Predicted top-10 items for user 0")
    print(f"  Top item: {items[0]}, score: {scores[0]:.4f}")

    # Test finding correct predictions
    test_user_dict = data_loader.test_user_dict

    correct = predictor.find_correct_predictions(test_user_dict, top_k=20)
    print(f"✓ Found {len(correct)} correct predictions")
    if correct:
        user_id, item_id, rank, score = correct[0]
        print(
            f"  Example: User {user_id}, Item {item_id}, Rank {rank}, Score {score:.4f}"
        )


def test_explainer_integration():
    """Test KGATExplainer integration."""
    print("\n=== Testing KGATExplainer Integration ===")

    # Create dummy components
    config = KGATConfig(
        embed_dim=64,
        relation_dim=64,
        aggregation_type="bi-interaction",
        conv_dim_list="[64,32,16]",
        mess_dropout="[0.1,0.1,0.1]",
    )

    model = KGAT(
        config=config,
        n_users=50,
        n_entities=100,
        n_relations=10,
    )
    model.eval()

    # Create simple KG
    kg_dict = {
        100: [(0, 0), (1, 1)],  # User 0 (adjusted) -> items
        101: [(2, 0), (3, 1)],  # User 1 (adjusted) -> items
        0: [(10, 2), (11, 3)],  # Item 0 -> entities
        1: [(10, 2), (12, 3)],  # Item 1 -> entities
    }

    kg_dict_by_relation = {
        0: [(100, 0), (101, 2)],
        1: [(100, 1), (101, 3)],
        2: [(0, 10), (1, 10)],
        3: [(0, 11), (1, 12)],
    }

    # Create a mock data_loader
    class MockDataLoader:
        def __init__(self):
            self.n_users = 50
            self.n_items = 50
            self.n_entities = 100
            self.test_batch_size = 10
            self.train_user_dict = {}
            self.test_user_dict = {}
            self.val_user_dict = {}

    data_loader = MockDataLoader()

    explainer = KGATExplainer(
        model=model,
        data_loader=data_loader,
        kg_dict=kg_dict,
        kg_dict_by_relation=kg_dict_by_relation,
        device=torch.device("cpu"),
        precompute_attention=True,
    )

    print(
        f"✓ Initialized explainer with {len(explainer.attention_scores)} attention scores"
    )

    # Test explaining a prediction
    explanation = explainer.explain_prediction(
        user_id=100, item_id=0, max_hops=2, max_paths=5
    )
    print(f"✓ Generated explanation for user 100, item 0")
    print(f"  Number of paths: {explanation['num_paths']}")
    if explanation["paths"]:
        print(f"  Best path score: {explanation['paths'][0]['total_score']:.4f}")


def main():
    """Run all tests."""
    logger = setup_logging()

    try:
        # Run individual tests
        test_attention_calculator()
        test_path_finder()
        test_predictor()
        test_explainer_integration()

        print("\n" + "=" * 50)
        print("✓ All tests passed successfully!")
        print("=" * 50)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
