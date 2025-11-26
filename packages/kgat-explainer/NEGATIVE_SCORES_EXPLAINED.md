# Understanding Negative Attention Scores in KGAT Explainer

## Overview

This document explains why attention scores can be negative, what they mean, and how to interpret them when analyzing KGAT model predictions.

## The Attention Score Formula

For an edge (head, relation, tail), the attention score is calculated as:

```python
r_embed = relation_embed[relation]
W_r = trans_M[relation]

h_embed = entity_embed[head]
t_embed = entity_embed[tail]

r_mul_h = h_embed @ W_r
r_mul_t = t_embed @ W_r

attention_score = sum(r_mul_t * tanh(r_mul_h + r_embed))
```

## Why Scores Can Be Negative

### 1. The tanh Function

The `tanh()` function has an output range of **[-1, 1]**:
- `tanh(large positive value)` → approaches +1
- `tanh(large negative value)` → approaches -1
- `tanh(0)` → 0

### 2. Embedding Directions

Neural network embeddings represent entities in a high-dimensional space where:
- **Similar** entities/concepts point in **similar directions**
- **Opposite** entities/concepts point in **opposite directions**

### 3. The Dot Product

The dot product `r_mul_t * tanh(...)` produces:
- **Positive values** when vectors point in the same direction
- **Negative values** when vectors point in opposite directions
- **Zero** when vectors are orthogonal

## Semantic Interpretation

### Positive Attention Scores

**Meaning**: The edge **supports** or **strengthens** the user-item connection.

**Examples**:
1. User likes Italian food → Restaurant serves Italian → Positive score
2. User prefers budget options → Item is inexpensive → Positive score
3. User lives in Seattle → Restaurant is in Seattle → Positive score

### Negative Attention Scores

**Meaning**: The edge **contradicts** or **weakens** the user-item connection.

**Examples**:
1. User dislikes spicy food → Restaurant is spicy → Negative score
2. User prefers quiet places → Restaurant is noisy → Negative score
3. User avoids seafood → Item contains seafood → Negative score

### Zero or Near-Zero Scores

**Meaning**: The edge is **neutral** or **irrelevant** to the recommendation.

**Examples**:
1. User prefers Italian food → Restaurant has WiFi → Near-zero score
2. User likes outdoors → Item color is blue → Near-zero score

## Real-World Example

### Scenario: Restaurant Recommendation

**User Profile** (embedded as):
- Likes: Italian food, budget-friendly, casual dining
- Dislikes: Expensive, formal, seafood

**Restaurant A** (high prediction score: 0.85):

**Path 1** (Score: +2.34):
```
User --[likes_cuisine]--> Italian --[cuisine_type]--> Restaurant A
  Edge 1: +1.20 (User → Italian)
  Edge 2: +1.14 (Italian → Restaurant A)
```
**Interpretation**: Strong positive - user likes Italian, restaurant is Italian

**Path 2** (Score: +0.87):
```
User --[price_preference]--> Budget --[price_category]--> Restaurant A
  Edge 1: +0.45 (User → Budget)
  Edge 2: +0.42 (Budget → Restaurant A)
```
**Interpretation**: Positive - user wants budget, restaurant is budget

**Path 3** (Score: -0.23):
```
User --[avoids]--> Seafood --[serves]--> Restaurant A
  Edge 1: +0.15 (User → Avoids Seafood)
  Edge 2: -0.38 (Avoids Seafood → Restaurant A serves some seafood)
```
**Interpretation**: Negative - restaurant serves *some* seafood dishes

**Overall**: Despite Path 3 being negative, Restaurant A is recommended because:
- Paths 1 and 2 are strongly positive (+2.34 + 0.87 = +3.21)
- Path 3 is weakly negative (-0.23)
- Net effect: +2.98 (strongly positive)

**Restaurant B** (low prediction score: 0.12):

**Path 1** (Score: -1.87):
```
User --[price_preference]--> Budget --[opposite]--> Expensive --[price_type]--> Restaurant B
  Edge 1: +0.45 (User → Budget)
  Edge 2: -1.20 (Budget ← → Expensive)
  Edge 3: -1.12 (Expensive → Restaurant B)
```
**Interpretation**: Strongly negative - user wants budget but restaurant is expensive

**Path 2** (Score: -0.65):
```
User --[atmosphere]--> Casual --[opposite]--> Formal --[style]--> Restaurant B
  Edge 1: +0.30 (User → Casual)
  Edge 2: -0.95 (Casual ← → Formal)
```
**Interpretation**: Negative - user prefers casual but restaurant is formal

**Overall**: Restaurant B is NOT recommended because:
- All paths are negative
- Strong contradictions with user preferences
- Net effect: -2.52 (strongly negative)

## Path Score Calculation

For a path with multiple edges:

```
path_score = edge1_score + edge2_score + edge3_score + ...
```

**Examples**:

1. **All Positive Path**:
   - Edge 1: +1.2
   - Edge 2: +0.8
   - Edge 3: +0.5
   - **Total: +2.5** (strong support)

2. **Mixed Path**:
   - Edge 1: +1.5
   - Edge 2: -0.3
   - Edge 3: +0.6
   - **Total: +1.8** (net positive, but some contradiction)

3. **All Negative Path**:
   - Edge 1: -0.9
   - Edge 2: -1.1
   - Edge 3: -0.4
   - **Total: -2.4** (strong contradiction)

## Common Misconceptions

### Misconception 1: "Negative scores are errors"

**FALSE**. Negative scores are normal and provide valuable information.

- They show which paths contradict the recommendation
- They explain why some items are NOT recommended
- They help understand the model's reasoning process

### Misconception 2: "Items with negative best paths shouldn't be recommended"

**FALSE**. The final prediction considers many factors:

1. **All paths**, not just the best one
2. **Direct embeddings** (user-item similarity)
3. **Multi-layer aggregation** from GNN layers
4. **Multiple neighborhood structures**

A single negative path can be outweighed by:
- Many weakly positive paths
- Strongly positive direct embeddings
- Other positive factors

### Misconception 3: "I should only look at positive paths"

**FALSE**. Negative paths are equally important:

- They show the **full picture** of the model's reasoning
- They explain **why** the model hesitated or lowered the score
- They identify **potential concerns** about the recommendation

### Misconception 4: "Negative scores mean the model is broken"

**FALSE**. This is the correct behavior of the attention mechanism:

- It's designed to capture both supporting and contradicting evidence
- It mirrors human decision-making (pros vs cons)
- It makes the model more interpretable and explainable

## Interpreting Explanations

### When Item IS Recommended (High Score)

**Look for**:
1. **Strongly positive paths**: Main supporting evidence
2. **Weakly negative paths**: Minor concerns that were outweighed
3. **Balance**: Positive evidence >> Negative evidence

**Example Interpretation**:
```
"Restaurant X was recommended because:
 - User loves Italian food and the restaurant is Italian (+2.3)
 - User prefers budget options and it's affordable (+0.9)
 - Despite serving some seafood which user dislikes (-0.2)
 → Net: Strong recommendation (+3.0)"
```

### When Item is NOT Recommended (Low Score)

**Look for**:
1. **Strongly negative paths**: Main reasons for rejection
2. **Weakly positive paths**: Minor supporting factors
3. **Balance**: Negative evidence >> Positive evidence

**Example Interpretation**:
```
"Restaurant Y was NOT recommended because:
 - User wants budget but restaurant is expensive (-1.9)
 - User prefers casual but restaurant is formal (-0.7)
 - Although it serves Italian which user likes (+0.4)
 → Net: Not recommended (-2.2)"
```

### When Item is MARGINALLY Recommended (Medium Score)

**Look for**:
1. **Mixed paths**: Both positive and negative
2. **Close balance**: Similar amounts of supporting/contradicting evidence
3. **Uncertain factors**: Many near-zero paths

**Example Interpretation**:
```
"Restaurant Z is a marginal recommendation because:
 - User likes the cuisine type (+1.2)
 - But it's slightly expensive for user's budget (-0.8)
 - And location is not ideal (-0.3)
 → Net: Weak recommendation (+0.1)"
```

## Practical Guidelines

### For Understanding Individual Recommendations

1. **Sum all path scores** to see net effect
2. **Identify dominant factors** (highest magnitude scores)
3. **Balance pros and cons** (positive vs negative paths)
4. **Consider path length** (longer paths are more indirect)

### For Debugging Model Behavior

1. **Unexpected high score**:
   - Look for hidden positive paths you didn't expect
   - Check if negative factors are being outweighed

2. **Unexpected low score**:
   - Look for strong negative paths
   - Check if expected positive paths are missing

3. **Inconsistent recommendations**:
   - Compare path patterns across similar users/items
   - Look for systematic biases in edge scores

### For Improving the Model

1. **Too many negative paths for good items**:
   - May indicate missing relations in the KG
   - Or need for better embeddings

2. **No negative paths for bad items**:
   - KG may lack negative/opposite relations
   - Model may be over-relying on positive signals

3. **All paths near zero**:
   - Embeddings may not be well-trained
   - KG structure may be too sparse

## Mathematical Note

The attention mechanism in KGAT is inspired by attention mechanisms in neural networks:

```
attention(h, r, t) = tanh(W_r @ h + r) · (W_r @ t)
```

This is similar to scaled dot-product attention but with:
- **Relation-specific transformations** (W_r)
- **Non-linear activation** (tanh)
- **Additive relation bias** (r)

The `tanh` ensures bounded outputs and allows:
- **Gradient flow** during training
- **Semantic similarity/opposition** in embeddings
- **Interpretable magnitudes** (always in [-∞, +∞] after sum, but typically in [-10, +10])

## Summary

**Key Takeaways**:

1. ✅ Negative attention scores are **normal and expected**
2. ✅ They represent **contradicting evidence** or **opposing factors**
3. ✅ They provide **valuable information** about model reasoning
4. ✅ Final predictions **combine all paths** and other factors
5. ✅ Both positive and negative paths are needed for **full interpretability**

**Remember**: Think of path explanations like a **decision scorecard**:
- Positive paths = reasons to recommend (pros)
- Negative paths = reasons not to recommend (cons)
- Final score = weighted sum of all evidence

This makes KGAT's recommendations more transparent, debuggable, and trustworthy.