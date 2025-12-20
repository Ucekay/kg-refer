from dataclasses import dataclass

from cyclopts import Parameter


@Parameter(name="*")
@dataclass
class LightKGConfig:
    dataset: str = "yelp"
    seed: int = 2020
    fix_relation_weights: bool = False  # If True, set all relation weights to 1 (equivalent to LightGCN)
