from dataclasses import dataclass
from pathlib import Path

from cyclopts import Parameter


@Parameter(name="*")
@dataclass
class CleanerConfig:
    input: Path = Path("input.json")
    "Input JSON file containing KG data (canon_kg.json format)"

    output: Path = Path("cleaned.json")
    "Output path for cleaned KG data"

    conflicts: Path = Path("conflicts.json")
    "Output path for conflicts (h,t pairs with multiple relations)"

    normalize: bool = False
    "Apply case normalization to relations and tails with variations"

    deduplicate: bool = False
    "Remove exact duplicate triplets (h, r, t)"

    find_conflicts: bool = False
    "Find and save (h, t) pairs with multiple relations"

    filter_terms: bool = False
    "Remove triplets containing filtered terms (Unknown, Cuisine, Food, Users, etc.)"

    filter_relations: bool = False
    "Remove triplets with specific relations (configured in code)"

    expand_entities: bool = False
    "Expand tail entities into multiple triplets (e.g., 'Cajun/Creole' → 'Cajun' and 'Creole')"

    replace_entities: bool = False
    "Replace tail entities regardless of relation (e.g., 'Food allergies' → 'food allergies')"

    replace_combinations: bool = False
    "Replace specific (relation, tail) combinations based on predefined rules"

    unify_relations: bool = False
    "Unify relations for tails with multiple relations based on tail unification rules"

    verbose: bool = False
    "Enable verbose logging"
