from .BaseDataModel import BaseDataModel
from models.db_schemas import Asset  # Ensure this is the SQLAlchemy model
from .enums.DataBaseEnums import DatabaseEnum
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

class AssetModel(BaseDataModel):

    def __init__(self, db_session: object):
        """
        In this pattern, db_client is the session factory (async_sessionmaker).
        """
        super().__init__(db_session=db_session)

    @classmethod
    async def create_instance(cls, db_session: AsyncSession):
        """
        Factory method to create an instance of AssetModel.
        Collection initialization is no longer needed for SQL.
        """
        return cls(db_session)
    
    async def create_asset(self, asset: Asset):
        """
        Adds the asset to the session. No internal commit—the route dependency 
        handles the transaction lifecycle.
        """
        self.db_session.add(asset)
        # Flush to DB so the object gets its ID assigned
        await self.db_session.flush()
        # Refresh ensures we have any DB-generated defaults (like created_at)
        await self.db_session.refresh(asset)
        return asset

    async def get_all_project_assets(self, asset_project_id: int, asset_type: str):
            """
            Retrieves all assets for a project using the shared session.
            """
            stmt = select(Asset).where(
                Asset.asset_project_id == asset_project_id,
                Asset.asset_type == asset_type
            )
            result = await self.db_session.execute(stmt)
            return result.scalars().all()

    async def get_asset_record(self, asset_project_id: int, asset_name: str):
            """
            Retrieves a single asset record using the shared session.
            """
            stmt = select(Asset).where(
                Asset.asset_project_id == asset_project_id,
                Asset.asset_name == asset_name
            )
            result = await self.db_session.execute(stmt)
            return result.scalar_one_or_none()