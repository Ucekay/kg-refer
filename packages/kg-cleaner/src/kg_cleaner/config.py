from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from cyclopts import Parameter


def add_timestamp_to_path(path: Path) -> Path:
    """Add timestamp to output file path."""
    if path.suffix:  # Has extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = path.stem
        suffix = path.suffix
        return path.parent / f"{stem}_{timestamp}{suffix}"
    else:  # No extension, treat as directory or add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(f"{path}_{timestamp}")


@Parameter(name="*")
@dataclass
class CleanerConfig:
    input: Path = Path("input.json")
    "Input JSON file containing KG data (canon_kg.json format)"

    output: Path = Path("cleaned.json")
    "Output path for cleaned KG data (timestamp will be automatically added)"

    conflicts: Path = Path("conflicts.json")
    "Output path for conflicts (h,t pairs with multiple relations) (timestamp will be automatically added)"

    def __post_init__(self):
        """Add timestamp to output paths after initialization."""
        # Only add timestamp if paths are not default values
        default_output = Path("cleaned.json")
        default_conflicts = Path("conflicts.json")
        
        if self.output != default_output:
            self.output = add_timestamp_to_path(self.output)
        
        if self.conflicts != default_conflicts:
            self.conflicts = add_timestamp_to_path(self.conflicts)

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

    replace_relations: bool = False
    "Replace relations regardless of head or tail (e.g., 'has ambience' → 'has atmosphere')"

    replace_combinations: bool = False
    "Replace specific (relation, tail) combinations based on predefined rules"

    unify_relations: bool = False
    "Unify relations for tails with multiple relations based on tail unification rules"

    normalize_accents: bool = False
    "Normalize accent marks in tail entities (é → e, à → a, etc.)"

    normalize_people: bool = False
    "Normalize tail entities containing 'Users', 'People', or 'users' to 'people'"

    verbose: bool = False
    "Enable verbose logging"
