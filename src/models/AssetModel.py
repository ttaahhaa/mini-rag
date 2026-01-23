from .BaseDataModel import BaseDataModel
from models.db_schemas import Asset  # Ensure this is the SQLAlchemy model
from .enums.DataBaseEnums import DatabaseEnum
from sqlalchemy.future import select

class AssetModel(BaseDataModel):

    def __init__(self, db_client: object):
        """
        In this pattern, db_client is the session factory (async_sessionmaker).
        """
        super().__init__(db_client=db_client)
        self.db_client = db_client

    @classmethod
    async def create_instance(cls, db_client: object):
        """
        Factory method to create an instance of AssetModel.
        Collection initialization is no longer needed for SQL.
        """
        instance = cls(db_client)
        return instance

    async def create_asset(self, asset: Asset):
        async with self.db_client() as session:
            async with session.begin():
                session.add(asset)
                await session.flush()
            # Transaction is auto-committed here
            await session.refresh(asset)
        return asset

    async def get_all_project_assets(self, asset_project_id: str, asset_type: str):
        """
        Retrieves all assets for a project based on project_id and asset_type.
        """
        async with self.db_client() as session:
            # We use select() and where() with multiple conditions separated by commas
            stmt = select(Asset).where(
                Asset.asset_project_id == asset_project_id,
                Asset.asset_type == asset_type
            )
            result = await session.execute(stmt)
            # .scalars().all() converts the result rows into a list of Asset objects
            records = result.scalars().all()
        return records

    async def get_asset_record(self, asset_project_id: str, asset_name: str):
        """
        Retrieves a single asset record by project ID and asset name.
        """
        async with self.db_client() as session:
            stmt = select(Asset).where(
                Asset.asset_project_id == asset_project_id,
                Asset.asset_name == asset_name
            )
            result = await session.execute(stmt)
            # scalar_one_or_none() is used when you expect 1 or 0 results
            record = result.scalar_one_or_none()
        return record