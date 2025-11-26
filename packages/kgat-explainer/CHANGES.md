# KGAT Explainer - Changes Log

## Refactoring: Reuse KGAT's Prediction Implementation

### Summary

KGATPredictor has been refactored to reuse KGAT's existing `evaluate` function instead of reimplementing prediction logic. This ensures complete consistency with KGAT's evaluation framework.

### Changes Made

#### 1. KGATPredictor (`predictor.py`)

**Before:**
```python
predictor = KGATPredictor(
    model=model,
    n_users=n_users,
    n_items=n_items,
    n_entities=n_entities,
    device=device
)
```

**After:**
```python
predictor = KGATPredictor(
    model=model,
    data_loader=data_loader,  # Now takes DataLoader instead of counts
    device=device
)
```

**Key Changes:**
- Constructor now accepts `data_loader` instead of individual counts
- `evaluate_metrics()` now calls KGAT's `evaluate()` function directly
- Removed custom metric calculation logic
- Added `get_all_predictions()` method that wraps KGAT's evaluate

#### 2. KGATExplainer (`explainer.py`)

**Before:**
```python
explainer = KGATExplainer(
    model=model,
    kg_dict=kg_dict,
    kg_dict_by_relation=kg_dict_by_relation,
    n_users=n_users,
    n_items=n_items,
    n_entities=n_entities,
    device=device
)
```

**After:**
```python
explainer = KGATExplainer(
    model=model,
    data_loader=data_loader,  # Now required
    kg_dict=kg_dict,
    kg_dict_by_relation=kg_dict_by_relation,
    device=device
)
```

**Key Changes:**
- Constructor now requires `data_loader` parameter
- Removes `n_users`, `n_items`, `n_entities` parameters (extracted from data_loader)
- Passes data_loader to KGATPredictor

#### 3. All Usage Sites Updated

- `explain.py`: Both `explain_correct_predictions()` and `explain_user_predictions()`
- `stats.py`: Both `compute_statistics()` and `evaluate_model()`
- `example_usage.py`: Main example updated
- `test_explainer.py`: Tests updated with mock DataLoader

### Benefits

1. **Consistency**: Predictions and metrics are now 100% identical to KGAT's evaluation
2. **Maintainability**: No duplicate prediction logic to maintain
3. **Correctness**: Uses battle-tested KGAT evaluation code
4. **Features**: Automatically inherits any improvements to KGAT's evaluate function

### API Impact

**Breaking Change**: Users must now pass `data_loader` to both `KGATPredictor` and `KGATExplainer`.

**Migration Guide:**

```python
# Old code
predictor = KGATPredictor(model, n_users, n_items, n_entities, device)
explainer = KGATExplainer(model, kg_dict, kg_dict_by_relation, 
                          n_users, n_items, n_entities, device)

# New code
predictor = KGATPredictor(model, data_loader, device)
explainer = KGATExplainer(model, data_loader, kg_dict, 
                          kg_dict_by_relation, device)
```

### Implementation Details

#### evaluate_metrics() Method

**Old Implementation** (Custom):
```python
def evaluate_metrics(self, test_user_dict, k_list):
    # Custom logic for computing precision, recall, NDCG
    # ~50 lines of metric calculation code
    return metrics
```

**New Implementation** (Reuses KGAT):
```python
def evaluate_metrics(self, test_user_dict, k_list, use_validation=False):
    # Direct call to KGAT's evaluate function
    _, metrics_dict = evaluate(
        self.model, self.data_loader, k_list, self.device, use_validation
    )
    return metrics_dict
```

#### New Method: get_all_predictions()

```python
def get_all_predictions(self, use_validation=False):
    """Get predictions for all users using KGAT's evaluate function.
    
    Returns:
        Tuple of (cf_scores, metrics_dict)
    """
    Ks = [20, 40, 60, 80, 100]
    cf_scores, metrics_dict = evaluate(
        self.model, self.data_loader, Ks, self.device, use_validation
    )
    return cf_scores, metrics_dict
```

This method provides access to the full prediction score matrix, useful for analysis.

### Testing

All tests pass with the new implementation:
- AttentionCalculator tests ✓
- PathFinder tests ✓
- KGATPredictor tests ✓ (updated with mock DataLoader)
- Integration tests ✓

### Compatibility

- **KGAT Package**: Fully compatible, reuses `kgat.training.evaluate`
- **Python**: No changes, still requires >=3.11
- **PyTorch**: No changes, still requires >=2.0.0

### Version

This refactoring is included in version 0.1.0.

### References

- KGAT evaluate function: `packages/kgat/src/kgat/training/evaluate.py`
- Original issue: "predict をあなたは実装したが kgat をインポートしてその実装の予測を再利用できないの"