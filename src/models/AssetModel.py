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
        
        :param db_client: The database client to interact with the database.
        :return: An initialized instance of AssetModel.
        """
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        """
        Initialize the asset collection in the database.
        This method checks if the collection exists, and if not, creates it
        along with the necessary indexes defined in the Asset schema.
        
        :param self: The instance of the AssetModel class.
        :return: None
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
        # 1. Convert Pydantic model to dict
        # dump using aliases so Mongo sees _id if present
        asset_dict = asset.model_dump(by_alias=True, exclude_unset=True)

        # ensure we don’t send an _id if it's None
        asset_dict.pop("_id", None)

        # 2. Insert into MongoDB
        result = await self.collection.insert_one(asset_dict)

        # 3. Attach MongoDB _id to the model
        asset.id = result.inserted_id  # <- use _id, not id

        # 4. Return updated asset
        return asset
    
    async def get_all_assets_by_project_id(self, project_id: str) -> list[Asset]:
        return await self.collection.find(
            {"asset_project_id": ObjectId(project_id)} if isinstance(project_id, str) else {"asset_project_id": project_id}
        ).to_list(length=None) # fetch all matching documents and None means no limit
      