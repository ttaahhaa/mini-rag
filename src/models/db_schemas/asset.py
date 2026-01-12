from typing import Optional

from bson.objectid import ObjectId
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime, timezone

class Asset(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_by_name=True,
        validate_by_alias=True,
    )

    id: Optional[ObjectId] = Field(default=None, alias="_id")  # maps Mongo _id <-> id
    asset_project_id: ObjectId
    asset_type: str = Field(..., min_length=1)
    asset_name: str = Field(..., min_length=1)
    asset_size: int = Field(gt=0, default=None)
    asset_config: Optional[dict] = Field(default=None)
    asset_pushed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def get_indexes(cls) ->list[dict]:
        return[
            {
                "key": [("asset_project_id", 1)], # ascending index
                "name": "asset_project_id_index_1", # index name
                "unique": False, # no uniqueness constraint
                
            },
            {
                "key": [("asset_name", 1)], # ascending index
                "name": "asset_name_index_1", # index name
                "unique": True, # uniqueness constraint because both project and asset name should be unique together
            }
        ]