
from sqlalchemy.ext.asyncio import AsyncSession
class BaseDataModel:
    def __init__(self, db_session: AsyncSession):
        self.db_session = db_session
        from helpers.config import get_settings
        self.app_settings = get_settings()
        