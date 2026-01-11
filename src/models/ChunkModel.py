from .BaseDataModel import BaseDataModel
from .db_schemas.data_chunks import DataChunkSchema
from .enums.DataBaseEnums import DatabaseEnum
from bson.objectid import ObjectId
from pymongo import InsertOne

class ChunkModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client=db_client)
        self.collection = db_client[DatabaseEnum.COLLECTION_CHUNK_NAME.value]
    
    async def create_chunk(self, chunk: DataChunkSchema) -> str | None:
        # Use aliases, do not write logical id
        doc = chunk.model_dump(
            by_alias=True,
            exclude={"id"},
            exclude_unset=True,
        )

        record = await self.collection.insert_one(doc)

        # Keep ObjectId in the model for internal use
        chunk.id = record.inserted_id

        if record.acknowledged:
            # Return string for API / callers
            return str(record.inserted_id)

        return None

    
    async def get_chunk(self, chunk_id: str):
        result = await self.collection.find_one({
            "_id": ObjectId(chunk_id) 
        })
        if not result:
            return None

        return DataChunkSchema(**result)

    async def insert_many_chunks(self, chunks: list, batch_size: int = 100) -> int:
        total_inserted = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            operations = [
            InsertOne(
                chunk.model_dump(
                    by_alias=True,        # use Mongo field names (e.g. "_id")
                    exclude={"id"},       # don't store logical id field
                    exclude_unset=True,
                )
            )
            for chunk in batch
        ]

            result = await self.collection.bulk_write(operations)
            # bulk_write result has bulk_api_result with nInserted.[web:74][web:76]
            total_inserted += result.bulk_api_result.get("nInserted", 0)

        return total_inserted


    async def get_all_chunks_in_a_project(self, project_id: ObjectId):
        records = await self.collection.find_many(
        {"chunk_project_id": project_id})
        if len(records) > 0:
            return records
        return None

    async def delete_a_chunk(self, chunk_id: str):
        result = await self.collection.find_one(chunk_id)
        if result:
            await self.collection.delete_one({
                "_id": chunk_id
            })
            return result
        return -1

    async def delete_chunks_by_project_id(self, project_id: ObjectId) -> int:
        result = await self.collection.delete_many({
            "chunk_project_id": project_id
        })
        return result.deleted_count
