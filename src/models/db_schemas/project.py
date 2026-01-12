from typing import Optional

from bson.objectid import ObjectId
from pydantic import BaseModel, Field, ConfigDict, field_validator


class ProjectSchema(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_by_name=True,
        validate_by_alias=True,
    )

    id: Optional[ObjectId] = Field(default=None, alias="_id")
    project_id: str = Field(..., min_length=1)

    @field_validator("project_id")
    @classmethod
    def project_id_must_not_be_empty(cls, value: str) -> str:
        # reject empty/whitespace
        if not value or not value.strip():
            raise ValueError("project ID must be a non-empty alphanumeric string")
        # enforce alphanumeric
        if not value.isalnum():
            raise ValueError("project ID must be a non-empty alphanumeric string")
        return value

    @classmethod
    def get_indexes(cls) ->list[dict]:
        return[
            {
                "key": [("project_id", 1)], # ascending index
                "name": "project_id_index_1", # index name
                "unique": True, # enforce uniqueness
                
            }
        ]