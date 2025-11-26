# KGAT Explainer - Usage Examples

This document provides complete command-line examples for running KGAT Explainer with a model trained using specific settings.

## Training Configuration

Assuming you trained a KGAT model with the following command:

```bash
uv run kgat --train \
  --data-name yelp \
  --data-dir "packages/kgat/datasets/" \
  --use-pretrain 0 \
  --n-epoch 1000 \
  --embed-dim 64 \
  --relation-dim 64 \
  --cf-batch-size 1024 \
  --kg-batch-size 1024 \
  --lr 0.0001 \
  --conv-dim-list "[64,32,16]" \
  --mess-dropout "[0.1,0.1,0.1]" \
  --laplacian-type random-walk \
  --aggregation-type bi-interaction \
  --kg-l2loss-lambda 1e-5 \
  --cf-l2loss-lambda 1e-5 \
  --evaluate-every 50
```

The trained model will be saved to:
```
packages/kgat/trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain0/
```

## Finding the Best Model

After training, check the logs or metrics files to find the best epoch. For example:
- `model_epoch300.pth` - if epoch 300 had the best validation performance

## Usage Examples

### 1. Explain Correct Predictions (Full Analysis)

```bash
uv run kgat-explainer --explain-correct \
  --model-path "packages/kgat/trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain0/model_epoch300.pth" \
  --data-name yelp \
  --data-dir "packages/kgat/datasets/" \
  --use-pretrain 0 \
  --embed-dim 64 \
  --relation-dim 64 \
  --laplacian-type random-walk \
  --aggregation-type bi-interaction \
  --conv-dim-list "[64,32,16]" \
  --mess-dropout "[0.1,0.1,0.1]" \
  --kg-l2loss-lambda 1e-5 \
  --cf-l2loss-lambda 1e-5 \
  --max-hops 3 \
  --max-paths-per-pair 10 \
  --top-k 100 \
  --sample-size 100 \
  --output-dir "output/explanations/" \
  --precompute-attention \
  --save-explanations \
  --verbose
```

**Output:**
- `output/explanations/yelp_explanations.json` - Detailed explanations for each correct prediction
- `output/explanations/yelp_statistics.json` - Path statistics and important edges

**What it does:**
1. Loads the trained model
2. Finds top-100 predictions for each test user
3. Identifies correct predictions (items in test set that ranked well)
4. Finds paths (up to 3 hops) connecting users to correctly predicted items
5. Computes attention scores for each edge in paths
6. Ranks paths by total attention score
7. Identifies most frequently used edges across all explanations

### 2. Quick Path Statistics (Lightweight)

```bash
uv run kgat-explainer --statistics \
  --model-path "packages/kgat/trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain0/model_epoch300.pth" \
  --data-name yelp \
  --data-dir "packages/kgat/datasets/" \
  --use-pretrain 0 \
  --embed-dim 64 \
  --relation-dim 64 \
  --laplacian-type random-walk \
  --aggregation-type bi-interaction \
  --conv-dim-list "[64,32,16]" \
  --mess-dropout "[0.1,0.1,0.1]" \
  --max-hops 3 \
  --sample-size 200
```

**Output (printed to console):**
```
=== Path Statistics ===
Total pairs sampled: 200
Pairs with paths: 165
Pairs without paths: 35
Path coverage: 82.50%
Average paths per pair: 8.34
Average path length: 2.1
Min path length: 1
Max path length: 3
Total paths found: 1376

=== Model Performance ===
K= 20: Precision=0.0542, Recall=0.0234, NDCG=0.0678
K= 40: Precision=0.0487, Recall=0.0421, NDCG=0.0823
...
```

**What it does:**
1. Samples 200 correct predictions
2. Computes path statistics (coverage, length distribution)
3. Evaluates model metrics using KGAT's evaluate function
4. Does NOT precompute attention (faster for statistics only)

### 3. Model Evaluation Only

```bash
uv run kgat-explainer --evaluate \
  --model-path "packages/kgat/trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain0/model_epoch300.pth" \
  --data-name yelp \
  --data-dir "packages/kgat/datasets/" \
  --use-pretrain 0 \
  --embed-dim 64 \
  --relation-dim 64 \
  --laplacian-type random-walk \
  --aggregation-type bi-interaction \
  --conv-dim-list "[64,32,16]" \
  --mess-dropout "[0.1,0.1,0.1]" \
  --Ks "[20,40,60,80,100]"
```

**Output:**
```
=== Evaluation Results ===
Dataset: yelp
Test users: 1237
Test interactions: 4567

Metrics:
  K= 20: Precision=0.0542, Recall=0.0234, NDCG=0.0678
  K= 40: Precision=0.0487, Recall=0.0421, NDCG=0.0823
  K= 60: Precision=0.0445, Recall=0.0576, NDCG=0.0912
  K= 80: Precision=0.0418, Recall=0.0723, NDCG=0.0987
  K=100: Precision=0.0398, Recall=0.0859, NDCG=0.1045
```

**What it does:**
- Uses KGAT's evaluate function to compute metrics
- Same metrics as KGAT's training evaluation
- Useful for verifying model performance before explanation

### 4. Explain Specific User's Predictions

```bash
uv run kgat-explainer --explain-user 0 \
  --model-path "packages/kgat/trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain0/model_epoch300.pth" \
  --data-name yelp \
  --data-dir "packages/kgat/datasets/" \
  --use-pretrain 0 \
  --embed-dim 64 \
  --relation-dim 64 \
  --laplacian-type random-walk \
  --aggregation-type bi-interaction \
  --conv-dim-list "[64,32,16]" \
  --mess-dropout "[0.1,0.1,0.1]" \
  --max-hops 3 \
  --top-k 20 \
  --max-paths-per-pair 5 \
  --output-dir "output/explanations/" \
  --save-explanations
```

**Output:**
```
=== Top 20 Predictions for User 0 ===

Rank 1: Item 1234 (score: 0.8542, paths: 5)
  Best path score: 2.3456
    Edge 1: 10050 --[0]--> 1234 (attention: 1.2345)
    Edge 2: 10050 --[2]--> 567 (attention: 0.8901)
    Edge 3: 567 --[5]--> 1234 (attention: 0.2210)

Rank 2: Item 5678 (score: 0.7823, paths: 3)
...
```

**File saved:** `output/explanations/yelp_user0_explanations.json`

**What it does:**
1. Predicts top-20 items for user 0
2. For each predicted item, finds up to 5 paths (max 3 hops)
3. Ranks paths by attention scores
4. Shows interpretation of why each item was recommended

### 5. GPU-Accelerated Explanation

```bash
uv run kgat-explainer --explain-correct \
  --model-path "packages/kgat/trained_model/KGAT/yelp/embed-dim64_relation-dim64_random-walk_bi-interaction_64-32-16_lr0.0001_pretrain0/model_epoch300.pth" \
  --data-name yelp \
  --data-dir "packages/kgat/datasets/" \
  --use-pretrain 0 \
  --embed-dim 64 \
  --relation-dim 64 \
  --laplacian-type random-walk \
  --aggregation-type bi-interaction \
  --conv-dim-list "[64,32,16]" \
  --mess-dropout "[0.1,0.1,0.1]" \
  --max-hops 3 \
  --use-cuda \
  --precompute-attention
```

**Note:** Adds `--use-cuda` flag to use GPU for:
- Attention score computation (10-100x faster)
- Model inference
- Batch predictions

## Configuration Matching

**CRITICAL:** The explainer configuration MUST match the training configuration for the following parameters:

| Parameter | Training Value | Explainer Flag |
|-----------|---------------|----------------|
| Data name | `yelp` | `--data-name yelp` |
| Use pretrain | `0` | `--use-pretrain 0` |
| Embed dim | `64` | `--embed-dim 64` |
| Relation dim | `64` | `--relation-dim 64` |
| Laplacian type | `random-walk` | `--laplacian-type random-walk` |
| Aggregation type | `bi-interaction` | `--aggregation-type bi-interaction` |
| Conv dims | `[64,32,16]` | `--conv-dim-list "[64,32,16]"` |
| Mess dropout | `[0.1,0.1,0.1]` | `--mess-dropout "[0.1,0.1,0.1]"` |

**Mismatched configurations will cause model loading errors or incorrect results.**

## Common Options Reference

### Required for All Modes
- `--model-path`: Path to trained model file
- `--data-name`: Dataset name (must match training)
- Model architecture flags (must match training)

### Path Finding Options
- `--max-hops`: Maximum path length (default: 3)
  - 1 hop: Direct user-item edges only
  - 2 hops: User -> Entity -> Item
  - 3 hops: User -> Entity -> Entity -> Item
  - Higher values = more paths but slower

- `--max-paths-per-pair`: Max paths to find per user-item pair (default: 10)
  - Lower values = faster but less complete explanations
  - Higher values = more comprehensive but slower

### Prediction Options
- `--top-k`: Consider top-K predictions (default: 100)
- `--min-rank`: Minimum rank to consider correct (default: 1)
- `--sample-size`: Number of pairs to process (default: all)

### Performance Options
- `--precompute-attention`: Precompute all attention scores (faster but uses more memory)
- `--use-cuda`: Use GPU if available
- `--verbose`: Enable detailed logging

### Output Options
- `--output-dir`: Directory for output files
- `--save-explanations`: Save results to JSON files

## Interpreting Results

### Explanation JSON Structure

```json
{
  "user_id": 10050,
  "item_id": 1234,
  "rank": 1,
  "prediction_score": 0.8542,
  "num_paths": 5,
  "paths": [
    {
      "path": [
        {
          "head": 10050,
          "relation": 0,
          "tail": 1234,
          "attention_score": 1.2345
        }
      ],
      "length": 1,
      "total_score": 1.2345,
      "avg_score": 1.2345
    }
  ]
}
```

### Understanding Attention Scores

**Attention scores can be positive or negative** - this is normal behavior:

- **Positive scores**: Edge contributes positively to the recommendation
  - The edge strengthens the connection between user and item
  - Example: User likes Italian food → Restaurant is Italian → Recommended
  
- **Negative scores**: Edge contributes negatively to the recommendation
  - The edge weakens or contradicts the recommendation
  - Example: User dislikes expensive → Restaurant is expensive → Not recommended
  - Negative edges still provide information about why a prediction was made

- **Higher magnitude**: Stronger influence on the prediction (regardless of sign)

- **Total score**: Sum of all edge scores in the path
  - Can be positive, negative, or zero
  - Negative total score means the path overall discourages the recommendation
  
- **Average score**: Total score / path length
  - Useful for comparing paths of different lengths
  - Normalize the impact of path length

**Why negative scores occur:**
The attention calculation is: `sum(r_mul_t * tanh(r_mul_h + r_embed))`
- `tanh()` output range is [-1, 1]
- When embeddings point in opposite directions, the dot product is negative
- This captures semantic opposition (e.g., user preferences vs. item attributes)

**Important:** Even if the best path has a negative score, the item can still be recommended because:
1. Other factors (direct embeddings, other paths) may be strongly positive
2. KGAT's final prediction combines multiple signals beyond individual paths
3. The path explanation shows *how* the KG influenced the decision, not *whether* to recommend

### Path Types

1. **Direct paths** (1 hop):
   - User directly connected to Item
   - Example: User rated/interacted with Item

2. **2-hop paths**:
   - User -> Entity -> Item
   - Example: User likes Restaurant -> Restaurant serves Cuisine -> Item is same Cuisine

3. **3-hop paths**:
   - User -> Entity -> Entity -> Item
   - Example: User likes Restaurant -> Restaurant in City -> City has Category -> Item in Category

## Troubleshooting

### "Model file not found"
- Check that the epoch number in the path matches the best model
- Verify the model directory exists

### "Shape mismatch" or "Unexpected key" errors
- Ensure all model architecture parameters match training configuration
- Check `--embed-dim`, `--relation-dim`, `--conv-dim-list`, etc.

### "Out of memory"
- Disable attention precomputation: remove `--precompute-attention`
- Reduce `--max-hops` (try 2 instead of 3)
- Reduce `--sample-size`
- Reduce `--max-paths-per-pair`

### "Very slow execution"
- Enable GPU: add `--use-cuda`
- Enable attention precomputation: add `--precompute-attention`
- Reduce `--max-hops`
- Reduce `--sample-size`

### "No paths found"
- This is normal - not all user-item pairs are connected
- Check path statistics with `--statistics` to see coverage
- Consider increasing `--max-hops`

## Python API Examples

If you prefer using the Python API directly:

```python
import torch
from kgat.config import KGATConfig
from kgat.core.kgat import KGAT
from kgat.data.dataloader import DataLoader
from kgat_explainer import KGATExplainer, KGATModelLoader

# Configuration
config = KGATConfig(
    data_name="yelp",
    data_dir="packages/kgat/datasets/",
    use_pretrain=0,
    embed_dim=64,
    relation_dim=64,
    aggregation_type="bi-interaction",
    conv_dim_list="[64,32,16]",
    mess_dropout="[0.1,0.1,0.1]",
)

# Load data
data_loader = DataLoader(config, logger)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_loader = KGATModelLoader(
    "packages/kgat/trained_model/.../model_epoch300.pth",
    device
)
model = model_loader.load_model(
    model_class=KGAT,
    config=config,
    n_users=data_loader.n_users,
    n_entities=data_loader.n_entities,
    n_relations=data_loader.n_relations,
    A_in=data_loader.A_in.to(device),
)

# Initialize explainer
kg_dict_by_relation = dict(data_loader.train_relation_dict)
explainer = KGATExplainer(
    model=model,
    data_loader=data_loader,
    kg_dict=data_loader.train_kg_dict,
    kg_dict_by_relation=kg_dict_by_relation,
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

# Get statistics
stats = explainer.get_path_statistics(
    test_user_dict=data_loader.test_user_dict,
    max_hops=3,
    sample_size=100,
)

print(f"Path coverage: {stats['path_coverage']:.2%}")
```
