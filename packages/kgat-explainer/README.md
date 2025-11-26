# KGAT Explainer

A comprehensive explanation framework for KGAT (Knowledge Graph Attention Network) models that provides path-based explanations for recommendations by analyzing attention scores on knowledge graph edges.

## Overview

KGAT Explainer helps you understand why a KGAT model makes specific recommendations by:

1. **Finding correct predictions**: Identifies user-item pairs where the model correctly predicted interactions
2. **Exploring connection paths**: Discovers l-hop paths connecting users to items through the knowledge graph
3. **Computing attention scores**: Calculates edge importance using KGAT's attention mechanism
4. **Providing explanations**: Generates human-interpretable explanations for recommendations

## Features

- **Efficient Attention Computation**: Precomputes attention scores for all edges in the knowledge graph
- **Flexible Path Finding**: BFS-based path exploration with configurable hop limits
- **Comprehensive Evaluation**: Built-in metrics (Precision, Recall, NDCG@K) for model performance
- **Batch Processing**: Efficient explanation generation for multiple user-item pairs
- **Statistical Analysis**: Path statistics and edge importance ranking

## Installation

This package is part of the kg-refer workspace. Install dependencies with:

```bash
uv sync
```

## Quick Start

### Command Line Interface

The easiest way to use KGAT Explainer is through the CLI:

```bash
# Explain correct predictions
uv run kgat-explainer --explain-correct \
    --model-path packages/kgat/trained_model/KGAT/yelp/.../model_epoch300.pth \
    --data-name yelp \
    --max-hops 3

# Compute path statistics
uv run kgat-explainer --statistics \
    --model-path path/to/model.pth \
    --sample-size 100

# Evaluate model performance
uv run kgat-explainer --evaluate \
    --model-path path/to/model.pth

# Explain predictions for a specific user
uv run kgat-explainer --explain-user 0 \
    --model-path path/to/model.pth \
    --top-k 10
```

### Python API

```python
import torch
from kgat.config import KGATConfig
from kgat.core.kgat import KGAT
from kgat.data.dataloader import DataLoader
from kgat_explainer import KGATExplainer, KGATModelLoader

# Load configuration and data
config = KGATConfig(data_name="yelp", data_dir="datasets/")
logger = logging.getLogger(__name__)
data_loader = DataLoader(config, logger)

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_loader = KGATModelLoader("path/to/model.pth", device)
model = model_loader.load_model(
    model_class=KGAT,
    config=config,
    n_users=data_loader.n_users,
    n_entities=data_loader.n_entities,
    n_relations=data_loader.n_relations,
    A_in=data_loader.A_in.to(device),
)

# Initialize explainer
explainer = KGATExplainer(
    model=model,
    data_loader=data_loader,
    kg_dict=data_loader.train_kg_dict,
    kg_dict_by_relation=data_loader.train_relation_dict,
    device=device,
    precompute_attention=True,
)

# Explain correct predictions
explanations = explainer.explain_correct_predictions(
    test_user_dict=data_loader.test_user_dict,
    max_hops=3,
    max_paths_per_pair=10,
    top_k=100,
)

# Get path statistics
stats = explainer.get_path_statistics(
    test_user_dict=data_loader.test_user_dict,
    max_hops=3,
)
print(f"Path coverage: {stats['path_coverage']:.2%}")
```

## Components

### 1. KGATExplainer

The main class that orchestrates all explanation functionality.

```python
explainer = KGATExplainer(
    model=model,                          # Trained KGAT model
    data_loader=data_loader,              # KGAT DataLoader
    kg_dict=kg_dict,                      # head -> [(tail, relation), ...]
    kg_dict_by_relation=kg_dict_by_rel,  # relation -> [(head, tail), ...]
    device=device,
    precompute_attention=True,            # Precompute all attention scores
)
```

#### Methods

- `explain_prediction(user_id, item_id, max_hops, max_paths)`: Explain a single prediction
- `explain_correct_predictions(test_user_dict, ...)`: Explain all correct predictions
- `explain_top_predictions(user_id, ...)`: Explain top-k predictions for a user
- `get_path_statistics(test_user_dict, ...)`: Get path statistics
- `find_most_important_edges(explanations, top_k)`: Find most important edges

### 2. AttentionCalculator

Computes attention scores for knowledge graph edges using KGAT's attention mechanism.

```python
calculator = AttentionCalculator(
    entity_user_embed=model.entity_user_embed,
    relation_embed=model.relation_embed,
    trans_M=model.trans_M,
    device=device,
)

# Calculate single edge attention
score = calculator.calculate_edge_attention(head, tail, relation)

# Precompute all attention scores
attention_scores = calculator.precompute_attention_scores_by_relation(
    kg_data_by_relation
)
```

### 3. PathFinder

Finds paths between users and items in the knowledge graph.

```python
path_finder = PathFinder(kg_dict=kg_dict, n_entities=n_entities)

# Find all paths within max_hops
paths = path_finder.find_paths(user_id, item_id, max_hops=3)

# Find shortest paths only
shortest = path_finder.find_shortest_paths(user_id, item_id, max_hops=3)

# Count paths without enumerating them
count = path_finder.count_paths(user_id, item_id, max_hops=3)
```

### 4. KGATPredictor

Handles prediction and evaluation of KGAT models.

```python
predictor = KGATPredictor(model, data_loader, device)

# Predict top-k items for a user
items, scores = predictor.predict_for_user(user_id, top_k=100)

# Find correct predictions
correct = predictor.find_correct_predictions(test_user_dict, top_k=100)

# Evaluate metrics (uses KGAT's evaluate function internally)
metrics = predictor.evaluate_metrics(test_user_dict, k_list=[20, 40, 60])
```

### 5. KGATModelLoader

Loads trained KGAT models from checkpoint files.

```python
loader = KGATModelLoader(model_path, device)

# Load model
model = loader.load_model(
    model_class=KGAT,
    config=config,
    n_users=n_users,
    n_entities=n_entities,
    n_relations=n_relations,
)

# Get model information
info = loader.get_model_info()
```

## Explanation Output Format

The `explain_prediction` method returns explanations in the following format:

```python
{
    "user_id": 12345,
    "item_id": 678,
    "num_paths": 5,
    "rank": 3,                      # If from correct predictions
    "prediction_score": 0.8542,     # If from predictions
    "paths": [
        {
            "path": [
                {
                    "head": 12345,
                    "relation": 0,
                    "tail": 999,
                    "attention_score": 0.234
                },
                {
                    "head": 999,
                    "relation": 5,
                    "tail": 678,
                    "attention_score": 0.567
                }
            ],
            "length": 2,
            "total_score": 0.801,
            "avg_score": 0.4005
        },
        # ... more paths
    ]
}
```

## Path Statistics

The `get_path_statistics` method provides comprehensive statistics:

```python
{
    "total_pairs": 1000,
    "pairs_with_paths": 850,
    "pairs_without_paths": 150,
    "path_coverage": 0.85,
    "avg_paths_per_pair": 12.5,
    "avg_path_length": 2.3,
    "min_path_length": 1,
    "max_path_length": 3,
    "total_paths": 10625
}
```

## Example Usage

See `example_usage.py` for a complete working example:

```bash
cd packages/kgat-explainer
uv run python example_usage.py
```

Or run the test suite:

```bash
uv run python packages/kgat-explainer/test_explainer.py
```

## Use Cases

### 1. Understanding Model Decisions

```python
# Explain why a specific item was recommended
explanation = explainer.explain_prediction(
    user_id=user_id,
    item_id=item_id,
    max_hops=3,
    max_paths=10
)

for path in explanation["paths"]:
    print(f"Path score: {path['total_score']:.4f}")
    for edge in path["path"]:
        print(f"  {edge['head']} --[{edge['relation']}]--> {edge['tail']}")
```

### 2. Finding Important Features

```python
# Identify most important edges across all explanations
explanations = explainer.explain_correct_predictions(test_user_dict)
important_edges = explainer.find_most_important_edges(explanations, top_k=20)

for h, r, t, avg_score, freq in important_edges:
    print(f"Edge ({h}, {r}, {t}): score={avg_score:.4f}, freq={freq}")
```

### 3. Debugging Model Performance

```python
# Compare predictions with actual paths
stats = explainer.get_path_statistics(test_user_dict, max_hops=3)

if stats['path_coverage'] < 0.5:
    print("Warning: Low path coverage - model may rely on non-path features")
```

### 4. Analyzing Recommendation Quality

```python
# Evaluate model and explain top predictions
metrics = predictor.evaluate_metrics(test_user_dict, k_list=[20, 40, 60])
print(f"Precision@20: {metrics[20]['precision']:.4f}")

# Get explanations for highly-ranked predictions
top_explanations = explainer.explain_top_predictions(
    user_id=user_id,
    max_hops=3,
    top_k=10
)
```

## Performance Considerations

### Memory Usage

- **Precomputing attention scores** requires O(E) memory where E is the number of edges
- For large graphs (>1M edges), consider:
  - Disabling precomputation: `precompute_attention=False`
  - Computing attention on-demand for specific paths

### Computation Time

- **Path finding** complexity depends on graph structure and max_hops
- For faster processing:
  - Limit `max_hops` (typically 2-3 is sufficient)
  - Set `max_paths` to limit enumeration
  - Use `find_shortest_paths()` instead of `find_paths()`

### GPU Acceleration

```python
# Use GPU for attention computation
device = torch.device("cuda:0")
explainer = KGATExplainer(..., device=device, precompute_attention=True)
```

## Technical Details

### Attention Score Calculation
### Understanding Attention Scores

The attention score for an edge (h, r, t) is calculated as:

```
score = sum(W_r * t_embed * tanh(W_r * h_embed + r_embed))
```

Where:
- `h_embed`: Embedding of head entity/user
- `t_embed`: Embedding of tail entity/user
- `r_embed`: Embedding of relation
- `W_r`: Transformation matrix for relation r

This is the **exact same attention mechanism** used during KGAT training.

**Important: Attention scores can be negative**

- **Positive scores**: The edge strengthens the user-item connection
- **Negative scores**: The edge weakens or contradicts the recommendation
- **Higher magnitude**: Stronger influence (regardless of sign)

Negative scores occur naturally when embeddings point in opposite directions (e.g., user preferences vs. item attributes). This is normal and provides information about why certain paths discourage recommendations. Even if the best path has a negative score, the item may still be recommended due to other factors (direct embeddings, other paths, etc.).

### Prediction and Evaluation

KGAT Explainer **reuses KGAT's `evaluate` function** for predictions and metric computation, ensuring:
- Identical prediction scores as KGAT
- Consistent metric calculations (Precision, Recall, NDCG@K)
- Same batch processing for efficiency

### Path Finding Algorithm

Uses breadth-first search (BFS) with cycle detection:
1. Start from user node
2. Explore neighbors through KG edges
3. Track visited nodes to prevent cycles
4. Stop at item node or max_hops reached

## Dependencies

- `torch >= 2.0.0`
- `numpy >= 1.24.0`
- `pandas >= 2.0.0`
- `scipy >= 1.10.0`
- `kgat` (from the same workspace)

## Citation

If you use this explainer in your research, please cite the original KGAT paper:

```bibtex
@inproceedings{wang2019kgat,
  title={KGAT: Knowledge Graph Attention Network for Recommendation},
  author={Wang, Xiang and He, Xiangnan and Cao, Yixin and Liu, Meng and Chua, Tat-Seng},
  booktitle={KDD},
  year={2019}
}
```

## CLI Reference

### Common Options

- `--model-path`: Path to trained KGAT model (.pth file)
- `--data-name`: Dataset name (default: "yelp")
- `--data-dir`: Directory containing dataset (default: "datasets/")
- `--max-hops`: Maximum number of hops for path finding (default: 3)
- `--use-cuda`: Use CUDA if available
- `--verbose`: Enable verbose logging

### Mode-Specific Options

**Explain Correct Predictions (`--explain-correct`)**:
- `--top-k`: Top-K predictions to consider (default: 100)
- `--max-paths-per-pair`: Maximum paths per user-item pair (default: 10)
- `--sample-size`: Number of predictions to explain (default: all)
- `--output-dir`: Directory to save explanations (default: "output/explanations/")

**Explain User Predictions (`--explain-user USER_ID`)**:
- `--top-k`: Number of top predictions to explain (default: 100)
- `--max-paths-per-pair`: Maximum paths per item (default: 10)

**Statistics (`--statistics`)**:
- `--sample-size`: Number of pairs to sample (default: 100)

**Evaluate (`--evaluate`)**:
- `--Ks`: List of K values for metrics (default: "[20,40,60,80,100]")

### Example Commands

```bash
# Full explanation with custom settings
uv run kgat-explainer --explain-correct \
    --model-path trained_model/model.pth \
    --data-name yelp \
    --max-hops 2 \
    --max-paths-per-pair 5 \
    --sample-size 50 \
    --use-cuda

# Quick statistics check
uv run kgat-explainer --statistics \
    --model-path trained_model/model.pth \
    --sample-size 200

# Explain top 20 items for user 42
uv run kgat-explainer --explain-user 42 \
    --model-path trained_model/model.pth \
    --top-k 20
```

## License

This package follows the same license as the parent kg-refer project.

## Contributing

Contributions are welcome! Please ensure:
- Code follows the existing style
- New features include tests
- Documentation is updated

## Frequently Asked Questions

### Why are some path scores negative?

**This is completely normal.** Attention scores can be positive or negative:

- **Positive scores**: The edge strengthens the user-item connection
- **Negative scores**: The edge weakens or contradicts the recommendation

The attention calculation is: `sum(r_mul_t * tanh(r_mul_h + r_embed))`

Since `tanh()` has range [-1, 1], and embeddings can point in opposite directions, the result can be negative. This captures semantic opposition (e.g., "user dislikes expensive" + "restaurant is expensive" = negative score).

**Even if the best path has a negative score, the item can still be recommended** because:
1. Other factors (direct embeddings, alternative paths) may be strongly positive
2. KGAT's final prediction combines multiple signals beyond individual paths
3. The path shows *how* the KG influenced the decision, not *whether* to recommend

Negative scores provide valuable information about why certain paths discourage recommendations while others encourage them.

### How should I interpret negative vs positive paths?

- **Positive paths**: Show supporting evidence for the recommendation
- **Negative paths**: Show contradicting evidence or negative associations
- **Both are useful**: They explain the full reasoning, not just the positive side

Think of it like: "User was recommended this item *despite* path A (negative) *because of* paths B and C (positive)."

## Troubleshooting

### "No paths found between user and item"

This is normal - not all user-item pairs are connected through the KG within max_hops. Check:
- Path statistics to see overall coverage
- Increase `max_hops` if needed
- Verify the KG contains expected edges

### "Out of memory during attention precomputation"

For large KGs:
```python
explainer = KGATExplainer(..., precompute_attention=False)
```

### "Slow path finding"

Reduce search space:
```python
paths = path_finder.find_paths(
    user_id, item_id, 
    max_hops=2,      # Reduce from 3
    max_paths=10     # Limit enumeration
)
```
