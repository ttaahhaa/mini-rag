from .BaseController import BaseController
from models import ResponseSignal
from models.db_schemas import ProjectSchema, DataChunkSchema
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json 

class NLPController(BaseController):
    def __init__(self, generation_client, embedding_client, vectordb_client):
        """
        Initializes the NLP Controller with necessary AI clients.
        :param generation_client: Client for text generation (LLM).
        :param embedding_client: Client for converting text to vectors.
        :param vectordb_client: Client for interacting with the Vector Database (e.g., Qdrant/Milvus).
        """
        super().__init__()
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.vectordb_client = vectordb_client

    def create_collection_name(self, project_id: str):
        """
        Standardizes the naming convention for vector database collections.
        Ensures each project has its own isolated space.
        """
        return f"Collection_{project_id}".strip()
    
    def reset_vector_db_collection(self, project: ProjectSchema):
        """
        Hard reset: Deletes the entire collection associated with a project.
        """
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)
    
    def get_vector_collection_info(self, project: ProjectSchema):
        """
        Retrieves status and stats about a project's collection.
        Uses a lambda trick in json.dumps to convert complex DB objects into serializable dictionaries.
        """
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)
        
        # We serialize/deserialize to ensure the output is a clean JSON/dict, 
        # handling cases where collection_info contains custom class instances.
        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
    def index_into_vector_db(self, project: ProjectSchema, 
                            chunks: List[DataChunkSchema], 
                            chunks_ids: List[int], 
                            do_reset: bool = False):
        
            collection_name = self.create_collection_name(project_id=project.project_id)
            texts = [c.chunk_text for c in chunks]
            metadata = [i.chunk_metadata for i in chunks]

            # PERFORMANCE BOOST: Use the new batch method instead of a loop
            vectors = self.embedding_client.embed_batch(
                texts=texts,
                document_type=DocumentTypeEnum.DOCUMENT.value
            )

            if not vectors:
                self.logger.error("Failed to generate batch embeddings")
                return False

            # Step 3 & 4 remain the same
            self.vectordb_client.create_collection(
                collection_name=collection_name,
                embedding_size=self.embedding_client.embedding_size,
                do_reset=do_reset
            )

            self.vectordb_client.insert_many(
                collection_name=collection_name,
                texts=texts,
                vectors=vectors,
                metadatas=metadata,
                record_ids=chunks_ids
            )

            return True

    def search_vector_db_collection(self, project: ProjectSchema, text: str, limit: int = 10):
        """
        Semantic Search logic:
        1. Converts user query into a vector.
        2. Uses vector similarity (cosine/euclidean) to find the 'nearest' chunks.
        """
        # step1: Identify the correct project collection
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: Convert user's question into a vector.
        # IMPORTANT: Use QUERY type here as some models (like Cohere) optimize 
        # vectors differently for queries vs stored documents.
        vector = self.embedding_client.embed_text(text=text,
                                                 document_type=DocumentTypeEnum.QUERY.value)

        if not vector or len(vector) == 0:
            return False

        # step3: Execute the search against the vector store
        results = self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=vector,
            limit=limit
        )

        if not results:
            return False

        # Transform results into a serializable format for the API response
        return json.loads(
            json.dumps(results, default=lambda x: x.__dict__)
        )