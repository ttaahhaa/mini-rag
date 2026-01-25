from .BaseDataModel import BaseDataModel
from models.db_schemas import DataChunk  # Ensure this is the SQLAlchemy model
from .enums.DataBaseEnums import DatabaseEnum
from sqlalchemy.future import select
from sqlalchemy import func, delete
from sqlalchemy.ext.asyncio import AsyncSession

class ChunkModel(BaseDataModel):

    def __init__(self, db_session: AsyncSession):
        super().__init__(db_session=db_session)

    @classmethod
    async def create_instance(cls, db_session: AsyncSession):
        """
        Factory method to create an instance using the injected session.
        """
        return cls(db_session)

    async def create_chunk(self, chunk: DataChunk):
            # Use self.db_session directly; no 'begin' or 'commit' needed
            self.db_session.add(chunk)
            await self.db_session.flush()
            await self.db_session.refresh(chunk)
            return chunk

    async def get_chunk(self, chunk_id: str):
            result = await self.db_session.execute(
                select(DataChunk).where(DataChunk.chunk_id == chunk_id)
            )
            return result.scalar_one_or_none()

    async def insert_many_chunks(self, chunks: list, batch_size: int = 100):
            """
            Inserts multiple chunks. Dependency handles the final commit.
            """
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                self.db_session.add_all(batch)
            
            # Flush to push to DB (useful if IDs are needed immediately)
            await self.db_session.flush()
            return len(chunks)

    async def delete_chunks_by_project_id(self, project_id: int):
            stmt = delete(DataChunk).where(DataChunk.chunk_project_id == project_id)
            result = await self.db_session.execute(stmt)
            return result.rowcount

    async def get_project_chunks(self, project_id: int, page_no: int = 1, page_size: int = 50):
            stmt = (
                select(DataChunk)
                .where(DataChunk.chunk_project_id == project_id)
                .offset((page_no - 1) * page_size)
                .limit(page_size)
            )
            result = await self.db_session.execute(stmt)
            return result.scalars().all()

    async def get_total_chunks_count(self, project_id: int):
            count_sql = (
                select(func.count(DataChunk.chunk_id))
                .where(DataChunk.chunk_project_id == project_id)
            )
            records_count = await self.db_session.execute(count_sql)
            return records_count.scalar()