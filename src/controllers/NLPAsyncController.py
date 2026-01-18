from .BaseAsyncController import BaseAsyncController
from models import ResponseSignal
from models.db_schemas import ProjectSchema, DataChunkSchema
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json 
import asyncio

class NLPAsyncController(BaseAsyncController):
    def __init__(self, generation_client, embedding_client, vectordb_client):
        """
        Initializes the Async NLP Controller.
        """
        super().__init__()
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.vectordb_client = vectordb_client

    def create_collection_name(self, project_id: str):
        return f"Collection_{project_id}".strip()
    
    async def reset_vector_db_collection(self, project: ProjectSchema):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.delete_collection(collection_name=collection_name)
    
    async def get_vector_collection_info(self, project: ProjectSchema):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = await self.vectordb_client.get_collection_info(collection_name=collection_name)
        
        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
    async def index_into_vector_db(self, project: ProjectSchema, 
                                   chunks: List[DataChunkSchema], 
                                   chunks_ids: List[int], 
                                   do_reset: bool = False):
        """
        Handles the full indexing pipeline asynchronously.
        """
        collection_name = self.create_collection_name(project_id=project.project_id)
        texts = [c.chunk_text for c in chunks]
        metadata = [i.chunk_metadata for i in chunks]

        # 1. Offload heavy embedding to a worker thread
        vectors = await asyncio.to_thread(
            self.embedding_client.embed_batch,
            texts=texts,
            document_type=DocumentTypeEnum.DOCUMENT.value
        )

        if not vectors:
            self.logger.error("Indexing failed: No vectors generated.")
            return False

        # 2. Create collection (Async database call)
        await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset
        )

        # 3. High-throughput async insertion
        result = await self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            vectors=vectors,
            metadatas=metadata,
            record_ids=chunks_ids
        )

        return result

    async def search_vector_db_collection(self, project: ProjectSchema, text: str, limit: int = 10):
        """
        Asynchronous semantic search.
        """
        collection_name = self.create_collection_name(project_id=project.project_id)

        # 1. Offload single-text embedding to a worker thread
        # This fixes the "TypeError: object list can't be used in 'await' expression"
        vector = await asyncio.to_thread(
            self.embedding_client.embed_text,
            text=text,
            document_type=DocumentTypeEnum.QUERY.value
        )

        if not vector or len(vector) == 0:
            return []

        # 2. Execute the async search against the vector store
        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=vector,
            limit=limit
        )

        return results if results else []