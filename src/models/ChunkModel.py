from .BaseDataModel import BaseDataModel
from models.db_schemas import DataChunk  # Ensure this is the SQLAlchemy model
from .enums.DataBaseEnums import DatabaseEnum
from sqlalchemy.future import select
from sqlalchemy import func, delete

class ChunkModel(BaseDataModel):

    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.db_client = db_client

    @classmethod
    async def create_instance(cls, db_client: object):
        """
        Factory method to create an instance of ChunkModel.
        Table initialization is handled by SQLAlchemy/Alembic, not here.
        """
        instance = cls(db_client)
        return instance

    async def create_chunk(self, chunk: DataChunk):
        async with self.db_client() as session:
            async with session.begin():
                session.add(chunk)
                await session.flush()
            await session.refresh(chunk)
        return chunk

    async def get_chunk(self, chunk_id: str):
        """
        Retrieves a specific chunk by its ID.
        """
        async with self.db_client() as session:
            # Note: Ensure DataChunk.chunk_id matches your SQL column name
            result = await session.execute(
                select(DataChunk).where(DataChunk.chunk_id == chunk_id)
            )
            chunk = result.scalar_one_or_none()
        return chunk

    async def insert_many_chunks(self, chunks: list, batch_size: int = 100):
        """
        Inserts multiple chunks efficiently using add_all.
        """
        async with self.db_client() as session:
            async with session.begin():
                for i in range(0, len(chunks), batch_size):
                    batch = chunks[i : i + batch_size]
                    session.add_all(batch)
            await session.commit()
        return len(chunks)

    async def delete_chunks_by_project_id(self, project_id: str):
        """
        Deletes all chunks associated with a specific project.
        """
        async with self.db_client() as session:
            stmt = delete(DataChunk).where(DataChunk.chunk_project_id == project_id)
            result = await session.execute(stmt)
            await session.commit()
        return result.rowcount  # Returns the number of deleted rows

    async def get_project_chunks(self, project_id: str, page_no: int = 1, page_size: int = 50):
        """
        Retrieves paginated chunks for a specific project.
        """
        async with self.db_client() as session:
            stmt = (
                select(DataChunk)
                .where(DataChunk.chunk_project_id == project_id)
                .offset((page_no - 1) * page_size)
                .limit(page_size)
            )
            result = await session.execute(stmt)
            records = result.scalars().all()
        return records

    async def get_total_chunks_count(self, project_id: str):
        """
        Counts total chunks for a specific project.
        """
        async with self.db_client() as session:
            count_sql = (
                select(func.count(DataChunk.chunk_id))
                .where(DataChunk.chunk_project_id == project_id)
            )
            records_count = await session.execute(count_sql)
            total_count = records_count.scalar()
        
        return total_count