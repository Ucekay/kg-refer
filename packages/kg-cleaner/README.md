# kg-cleaner

Knowledge Graph cleaning and normalization utilities for processing EDC output.

## Features

- **Term Filtering**: Removes triplets containing unwanted terms (Unknown, Cuisine, Food, Users, etc.)
- **Relation Filtering**: Removes triplets with specific relations (e.g., "appreciates", "likes")
- **Entity Expansion**: Expands one tail entity into multiple entities (e.g., "Cajun/Creole" → "Cajun" and "Creole")
- **Entity Replacement**: Replace tail entities regardless of relation (e.g., "Food allergies" → "food allergies")
- **Combination Replacement**: Replace specific (relation, tail) combinations based on predefined rules
- **Relation Unification**: Unifies relations when the same tail appears with multiple relations in one iid, keeping only the preferred relation
- **Smart Case Normalization**: Normalizes relations and tail entities to lowercase only when case variations exist (not a blanket lowercase conversion)
- **Duplicate Removal**: Removes exact duplicate triplets (h, r, t)
- **Conflict Detection**: Identifies (head, tail) pairs with multiple relations

## Installation

```bash
# Install from the kg-refer workspace root
uv pip install -e packages/kg-cleaner

# Or install directly in the package directory
cd packages/kg-cleaner
uv pip install -e .
```

## CLI Usage

### Basic Usage

**Note:** All features are disabled by default. You must explicitly enable at least one feature.

```bash
# Enable all features
kg-cleaner --input input.json --output cleaned.json --conflicts conflicts.json \
  --filter-terms --filter-relations --expand-entities --replace-entities --replace-combinations --unify-relations --normalize --deduplicate --find-conflicts

# Shorter syntax
kg-cleaner input.json output.json conflicts.json \
  --filter-terms --filter-relations --expand-entities --replace-entities --replace-combinations --unify-relations --normalize --deduplicate --find-conflicts
```

### Individual Features

Each cleaning feature must be explicitly enabled:

```bash
# Only filter unwanted terms
kg-cleaner input.json output.json conflicts.json --filter-terms

# Only filter specific relations
kg-cleaner input.json output.json conflicts.json --filter-relations

# Only expand entities
kg-cleaner input.json output.json conflicts.json --expand-entities

# Only replace entities
kg-cleaner input.json output.json conflicts.json --replace-entities

# Only replace combinations
kg-cleaner input.json output.json conflicts.json --replace-combinations

# Only unify relations
kg-cleaner input.json output.json conflicts.json --unify-relations

# Only normalize
kg-cleaner input.json output.json conflicts.json --normalize

# Only deduplicate
kg-cleaner input.json output.json conflicts.json --deduplicate

# Only find conflicts
kg-cleaner input.json output.json conflicts.json --find-conflicts

# Combine features as needed
kg-cleaner input.json output.json conflicts.json \
  --filter-terms --filter-relations --expand-entities --replace-entities --replace-combinations --unify-relations --normalize --deduplicate
```

### With Verbose Logging

```bash
kg-cleaner input.json output.json conflicts.json \
  --filter-terms --filter-relations --expand-entities --replace-entities --replace-combinations --unify-relations --normalize --deduplicate --verbose
```

### Help

```bash
kg-cleaner --help
```

### Configuration Parameters

All parameters can be set via command line:

**Required:**
- `--input`: Input JSON file containing KG data (canon_kg.json format)
- `--output`: Output path for cleaned KG data
- `--conflicts`: Output path for conflicts JSON

**Feature Flags (default: disabled - must be explicitly enabled):**
- `--filter-terms`: Remove triplets containing filtered terms (Unknown, Cuisine, Food, Users, dishes, unspecified, not specified)
- `--filter-relations`: Remove triplets with specific relations (configured in code)
- `--expand-entities`: Expand tail entities into multiple triplets (e.g., "Cajun/Creole" → "Cajun" and "Creole")
- `--replace-entities`: Replace tail entities regardless of relation (e.g., "Food allergies" → "food allergies")
- `--replace-combinations`: Replace specific (relation, tail) combinations based on predefined rules (configured in code)
- `--unify-relations`: Unify relations for tails with multiple relations based on tail unification rules (configured in code)
- `--normalize`: Apply case normalization to relations and tails with variations
- `--deduplicate`: Remove exact duplicate triplets (h, r, t)
- `--find-conflicts`: Find and save (h, t) pairs with multiple relations

**Other:**
- `--verbose`: Enable verbose logging (flag)

## Input Format

The input should be a JSON file with the following structure (same as EDC's `canon_kg.json`):

```json
[
  {
    "iid": 1,
    "triplets": [
      ["entity1", "relation1", "entity2"],
      ["entity1", "Relation1", "Entity2"],
      ["entity3", "relation2", "entity4"]
    ]
  },
  {
    "iid": 2,
    "triplets": [
      ["entity5", "relation3", "entity6"]
    ]
  }
]
```

## Output Format

### Cleaned Data

The cleaned output maintains the same structure as the input:

```json
[
  {
    "iid": 1,
    "triplets": [
      ["entity1", "relation1", "entity2"],
      ["entity3", "relation2", "entity4"]
    ]
  },
  {
    "iid": 2,
    "triplets": [
      ["entity5", "relation3", "entity6"]
    ]
  }
]
```

### Conflicts Data

Conflicts are saved in the same format as the input, with each (h, t) pair that has multiple relations saved as a separate entry. The `iid` is set to the head entity:

```json
[
  {
    "iid": "restaurant",
    "triplets": [
      ["restaurant", "located in", "Phoenix"],
      ["restaurant", "part of", "Phoenix"]
    ]
  },
  {
    "iid": "menu",
    "triplets": [
      ["menu", "has", "pizza"],
      ["menu", "includes", "pizza"]
    ]
  }
]
```

## How It Works

Features must be explicitly enabled via command-line flags:

1. **Term Filtering** (`--filter-terms`): Removes triplets where any field (head, relation, or tail) contains unwanted terms that are typically noise or placeholder values (Unknown, Cuisine, Food, Users, dishes, unspecified, not specified, etc.).

2. **Relation Filtering** (`--filter-relations`): Removes triplets with specific relations. Useful for removing unwanted relation types like "appreciates", "likes", etc. Relations to filter are configured in the code via the `FILTERED_RELATIONS` array.

3. **Entity Expansion** (`--expand-entities`): Expands one tail entity into multiple tail entities, creating multiple triplets. Useful for handling compound entities like "Cajun/Creole" → creates two separate triplets with "Cajun" and "Creole". Rules are configured in the code via the `ENTITY_EXPANSION_RULES` dictionary.

4. **Entity Replacement** (`--replace-entities`): Replaces tail entities regardless of relation. Useful for normalizing case, fixing typos, or standardizing terminology (e.g., "Food allergies" → "food allergies"). Rules are configured in the code via the `ENTITY_REPLACEMENTS` dictionary.

5. **Combination Replacement** (`--replace-combinations`): Replaces specific (relation, tail) combinations based on predefined rules. Rules are configured in the code via the `RELATION_TAIL_REPLACEMENTS` array. Example: replace `("serves", "Italian cuisine")` with `("serves", "Italian")`.

6. **Relation Unification** (`--unify-relations`): When the same tail appears with multiple relations in one iid, and a preferred (relation, tail) combination is specified in the rules, removes all other relations for that tail and keeps only the preferred one. Rules are configured in the code via the `TAIL_RELATION_UNIFICATION_RULES` dictionary. Example: if tail "big portions" appears with both "has feature" and "serves", and the rule specifies "serves" as preferred, removes "has feature" and keeps only "serves".

7. **Normalization** (`--normalize`): Analyzes all relations and tail entities. If a value has multiple case variations (e.g., "Beer" and "beer"), it normalizes them to lowercase. Values with consistent casing are left unchanged.

8. **Deduplication** (`--deduplicate`): Removes exact duplicate triplets (h, r, t).

9. **Conflict Detection** (`--find-conflicts`): Identifies cases where the same (head, tail) pair has multiple different relations, which may indicate data quality issues or legitimate multi-relation scenarios.

## Example Workflow

```bash
# Process EDC output with all features
kg-cleaner \
  --input packages/edc/output/yelp_async/iter0/canon_kg.json \
  --output output/cleaned_kg.json \
  --conflicts output/conflicts.json \
  --filter-terms \
  --filter-relations \
  --expand-entities \
  --replace-entities \
  --replace-combinations \
  --unify-relations \
  --normalize \
  --deduplicate \
  --find-conflicts \
  --verbose
```

## Python API

You can also use kg-cleaner programmatically:

```python
from kg_cleaner import KGCleaner
from pathlib import Path

cleaner = KGCleaner()
cleaner.clean_file(
    input_path=Path("input.json"),
    output_path=Path("cleaned.json"),
    conflicts_path=Path("conflicts.json"),
    filter_terms=True,
    filter_relations=True,
    expand_entities=True,
    replace_entities=True,
    replace_combinations=True,
    unify_relations=True,
    normalize=True,
    deduplicate=True,
    find_conflicts=True
)

# Or use the data directly
import json

with open("input.json") as f:
    data = json.load(f)

cleaned_data, conflicts = cleaner.clean_kg_data(
    data,
    filter_terms=True,
    filter_relations=True,
    expand_entities=True,
    replace_entities=True,
    replace_combinations=True,
    unify_relations=True,
    normalize=True,
    deduplicate=True,
    find_conflicts=True
)
```

### Configuring Filtering and Replacement Rules

Edit `kg-cleaner/src/kg_cleaner/cleaner.py` to configure filtering and replacement rules:

#### Filter Specific Relations

```python
# Relations to filter out (remove triplets with these relations)
FILTERED_RELATIONS = [
    "appreciates",
    "likes",
    "prefers",
    # Add more relations to filter here
]
```

#### Entity Expansion Rules

```python
# Entity expansion rules: Replace one tail entity with multiple tail entities
ENTITY_EXPANSION_RULES = {
    # Example: "Cajun/Creole" will create two triplets with "Cajun" and "Creole"
    "Cajun/Creole": ["Cajun", "Creole"],
    "Italian/French": ["Italian", "French"],
    # Add more expansion rules as needed
}
```

**Example:**
- Input: `["restaurant", "serves", "Cajun/Creole"]`
- Output: 
  - `["restaurant", "serves", "Cajun"]`
  - `["restaurant", "serves", "Creole"]`

#### Entity Replacement Rules

```python
# Entity-only replacement rules: Replace tail entity regardless of relation
ENTITY_REPLACEMENTS = {
    # Normalize case
    "Food allergies": "food allergies",
    "Intolerances": "intolerances",
    "Dietary restrictions": "dietary restrictions",
    "Healthy foods": "healthy food",
    
    # Fix common variations
    "Thai dishes": "Thai",
    "Mexican food": "Mexican",
    # Add more entity replacements as needed
}
```

**Example:**
- Input: `["restaurant", "caters to", "Food allergies"]`
- Output: `["restaurant", "caters to", "food allergies"]`

#### Replacement Rules for Combinations

```python
# Replacement rules for (relation, tail) combinations
RELATION_TAIL_REPLACEMENTS = [
    # Format: (old_relation, old_tail, new_relation, new_tail)
    # Use None to keep the original value
    ("serves", "Italian cuisine", "serves", "Italian"),
    ("serves", "Italian food", "serves", "Italian"),
    ("serves", "French cuisine", "serves", "French"),
    ("has feature", "big portions", None, "large servings"),  # Keep relation, change tail
    # Add more rules as needed
]
```

#### Relation Unification Rules

```python
# Tail-based relation unification rules
# When the same tail appears with multiple relations in one iid,
# if the specified (relation, tail) combination exists, keep only that one.
TAIL_RELATION_UNIFICATION_RULES = {
    # Format: {tail: preferred_relation}
    "big portions": "serves",
    "desserts": "serves",
    "Italian": "serves",
    "Italian cuisine": "serves",
    "Mexican": "serves",
    "beach hut": "is a",
    "dinner theatre": "is a",
    "families": "caters to",
    "business travelers": "caters to",
    "affordable": "price range",
    # Add more unification rules as needed
}
```

**Example:**
- Input (in one iid):
  - `["restaurant", "has feature", "big portions"]`
  - `["restaurant", "serves", "big portions"]`
- Output: `["restaurant", "serves", "big portions"]` (removed "has feature")

**How it works:**
1. Within each iid, identifies tails that appear with multiple relations
2. If a tail matches a unification rule AND the preferred relation exists
3. Removes all other relations for that tail, keeping only the preferred one

This is useful for resolving semantic conflicts where the same concept is expressed with different relations, and you want to standardize to the most appropriate relation.

## Statistics

After cleaning, kg-cleaner reports:
- Total triplets processed
- Number of triplets filtered (by term)
- Number of triplets filtered (by relation)
- Number of entities expanded
- Number of entities replaced
- Number of combinations replaced
- Number of relations unified
- Number of normalizations applied
- Number of duplicates removed
- Number of conflicts found

Example output:
```
2024-01-15 10:30:00 - kg_cleaner.cleaner - INFO - Processing 15234 total triplets
2024-01-15 10:30:01 - kg_cleaner.cleaner - INFO - Removed 156 triplets with filtered terms
2024-01-15 10:30:01 - kg_cleaner.cleaner - INFO - Removed 43 triplets with filtered relations
2024-01-15 10:30:01 - kg_cleaner.cleaner - INFO - Expanded 12 tail entities into multiple triplets
2024-01-15 10:30:01 - kg_cleaner.cleaner - INFO - Replaced 34 tail entities
2024-01-15 10:30:01 - kg_cleaner.cleaner - INFO - Replaced 87 relation-tail combinations
2024-01-15 10:30:02 - kg_cleaner.cleaner - INFO - Found 342 relations and 128 tails to normalize
2024-01-15 10:30:02 - kg_cleaner.cleaner - INFO - Removed 89 duplicate triplets
2024-01-15 10:30:02 - kg_cleaner.cleaner - INFO - Found 23 (h, t) pairs with multiple relations
2024-01-15 10:30:03 - kg_cleaner.cleaner - INFO - Saving cleaned data to output/cleaned_kg.json
2024-01-15 10:30:03 - kg_cleaner.cleaner - INFO - Saving 23 conflicts to output/conflicts.json
2024-01-15 10:30:03 - kg_cleaner.cleaner - INFO - Cleaning complete!
2024-01-15 10:30:03 - kg_cleaner.cleaner - INFO -   Total triplets: 15234
2024-01-15 10:30:03 - kg_cleaner.cleaner - INFO -   Filtered: 156
2024-01-15 10:30:03 - kg_cleaner.cleaner - INFO -   Filtered relations: 43
2024-01-15 10:30:03 - kg_cleaner.cleaner - INFO -   Entities expanded: 12
2024-01-15 10:30:03 - kg_cleaner.cleaner - INFO -   Entities replaced: 34
2024-01-15 10:30:03 - kg_cleaner.cleaner - INFO -   Combinations replaced: 87
2024-01-15 10:30:03 - kg_cleaner.cleaner - INFO -   Relations unified: 45
2024-01-15 10:30:03 - kg_cleaner.cleaner - INFO -   Normalized: 470
2024-01-15 10:30:03 - kg_cleaner.cleaner - INFO -   Duplicates removed: 89
2024-01-15 10:30:03 - kg_cleaner.cleaner - INFO -   Conflicts found: 23
```

## License

MIT