from typing import Optional

from bson.objectid import ObjectId
from pydantic import BaseModel, Field, ConfigDict, field_validator


class DataChunkSchema(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_by_name=True,
        validate_by_alias=True,
    )

    id: Optional[ObjectId] = Field(default=None, alias="_id")  # maps Mongo _id <-> id
    chunk_text: str = Field(..., min_length=1)
    chunk_metadata: dict
    chunk_order: int = Field(..., gt=0)
    chunk_project_id: ObjectId
    chunk_asset_id: Optional[ObjectId] = None

    @field_validator("chunk_text")
    @classmethod
    def text_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Chunk text must not be empty")
        return value

    @field_validator("chunk_metadata")
    @classmethod
    def metadata_must_be_dict(cls, value: dict) -> dict:
        if not isinstance(value, dict):
            raise ValueError("Chunk metadata must be a dictionary")
        return value

    @field_validator("chunk_order")
    @classmethod
    def order_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Chunk order must be a positive integer")
        return value
    
    @classmethod
    def get_indexes(cls) ->list[dict]:
        return[
            {
                "key": [("chunk_project_id", 1)], # ascending index
                "name": "chunk_project_id_index_1", # index name
                "unique": False, # no uniqueness constraint
                
            }
        ]

class RettrievedDocument(BaseModel):
    text: str
    score: float