from .BaseDataModel import BaseDataModel
from .db_schemas.asset import Asset
from .enums.DataBaseEnums import DatabaseEnum
from bson.objectid import ObjectId

class AssetModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client)
        self.collection = self.db_client[DatabaseEnum.COLLECTION_ASSET_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        """
        Factory method to create an instance of AssetModel and initialize the collection.
        """
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        """
        Ensures the collection and indexes exist.
        """
        all_collections = await self.db_client.list_collection_names()
        if DatabaseEnum.COLLECTION_ASSET_NAME.value not in all_collections:
            self.collection = self.db_client[DatabaseEnum.COLLECTION_ASSET_NAME.value]
            indexes = Asset.get_indexes()
            for index in indexes:
                await self.collection.create_index(
                    index["key"],
                    name=index["name"],
                    unique=index["unique"],
                )
    
    async def create_asset(self, asset: Asset):
        """
        Saves a new asset record to MongoDB.
        """
        asset_dict = asset.model_dump(by_alias=True, exclude_unset=True)
        asset_dict.pop("_id", None)
        result = await self.collection.insert_one(asset_dict)
        asset.id = result.inserted_id 
        return asset
    
    async def get_all_project_assets(self, project_id: str, asset_type: str) -> list[Asset]:
        """
        Retrieves all assets for a project and maps them to Pydantic objects.
        """
        query = {
            "asset_project_id": ObjectId(project_id) if isinstance(project_id, str) else project_id,
            "asset_type": asset_type
        }
        
        cursor = self.collection.find(query)
        assets_dicts = await cursor.to_list(length=None) 
        
        # Mapping to Pydantic Asset objects for dot notation access
        return [Asset(**asset_dict) for asset_dict in assets_dicts]