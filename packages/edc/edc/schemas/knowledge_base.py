from pydantic import BaseModel, Field


class Triplet(BaseModel):
    h: str = Field(description="Head entity of the triplet")
    r: str = Field(description="Relation of the triplet")
    t: str = Field(description="Tail entity of the triplet")


class KnowledgeBase(BaseModel):
    triplets: list[Triplet] = Field(description="List of extracted triplets")


#
