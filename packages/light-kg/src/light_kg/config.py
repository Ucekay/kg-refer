from dataclasses import dataclass

from cyclopts import Parameter


@Parameter(name="*")
@dataclass
class LightKGConfig:
    dataset: str = "yelp"
    seed: int = 2020
