import uuid
import logging
import asyncio
from qdrant_client import AsyncQdrantClient, models
from ..VectorDBInterfaceAsync import VectorDBInterfaceAsync
from ..VectorDBEnums import DistanceMetricEnums
from models.db_schemas import RettrievedDocument
from typing import List

class QDrantAsyncProvider(VectorDBInterfaceAsync):
    """
    QDrantAsyncProvider (v2): An optimized, asynchronous implementation of the Qdrant 
    Vector Database provider for the mini-RAG project.
    
    This class utilizes 'AsyncQdrantClient' to handle concurrent database operations,
    making it suitable for high-traffic environments (tens of requests per second).
    It also includes scalability optimizations like sharding and on-disk storage.
    """

    def __init__(self, db_path: str, url: str = None, api_key: str = None,
                 distance_metric: str = DistanceMetricEnums.COSINE.value):
        """
        Initializes the asynchronous provider with configuration settings.

        Args:
            db_path (str): Local path for embedded file storage (used if url is None).
            url (str, optional): Connection URL for remote Qdrant servers/clusters.
            api_key (str, optional): Security key for authenticated remote servers.
            distance_metric (str): Similarity calculation method (Cosine, Euclidean, or Dot).
        """
        self.db_path = db_path
        self.url = url
        self.api_key = api_key
        self.client = None
        self.logger = logging.getLogger(__name__)
        
        # Mapping metric strings from Enums to internal Qdrant distance models
        metric_map = {
            DistanceMetricEnums.COSINE.value: models.Distance.COSINE,
            DistanceMetricEnums.EUCLIDEAN.value: models.Distance.EUCLID,
            DistanceMetricEnums.DOT.value: models.Distance.DOT
        }
        self.distance_metric = metric_map.get(distance_metric, models.Distance.COSINE)

    async def _ensure_connected(self):
        """Internal helper to ensure client is connected before operations."""
        if self.client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

    async def connect(self):
        """Initializes the client with connection pooling settings."""
        if self.url:
            # For large scale, we specify limits and timeouts
            self.client = AsyncQdrantClient(
                url=self.url, 
                api_key=self.api_key,
                timeout=60
            )
        else:
            self.client = AsyncQdrantClient(path=self.db_path)
        self.logger.info("Async Qdrant client connected.")

    async def disconnect(self):
        """Ensures the underlying HTTP client is closed."""
        if self.client:
            await self.client.close()
            self.client = None
            self.logger.info("Async Qdrant client closed.")

    async def is_collection_exists(self, collection_name: str) -> bool:
        """
        Asynchronously checks if a collection exists.
        Returns: True if exists, False otherwise.
        """
        await self._ensure_connected()
        try:
            return await self.client.collection_exists(collection_name)
        except Exception as e:
            self.logger.error(f"Error checking collection existence: {e}")
            return False

    async def list_all_collections(self) -> list:
        """
        Retrieves a list of all existing collection names in the database.
        """
        await self._ensure_connected()
        try:
            collections_response = await self.client.get_collections()
            return [col.name for col in collections_response.collections]
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []

    async def get_collection_info(self, collection_name: str):
        """
        Retrieves technical metadata about a specific collection.
        Returns: A dictionary/object containing collection details or None if failed.
        """
        await self._ensure_connected()
        try:
            return await self.client.get_collection(collection_name=collection_name)
        except Exception as e:
            self.logger.error(f"Error retrieving info for '{collection_name}': {e}")
            return None

    async def delete_collection(self, collection_name: str):
        """
        Permanently deletes a collection and its data.
        """
        await self._ensure_connected()
        try:
            if await self.is_collection_exists(collection_name):
                await self.client.delete_collection(collection_name=collection_name)
                self.logger.info(f"Successfully deleted collection: {collection_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False

    async def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False):
        """
        Creates a collection with high-scale optimizations.
        
        Scalability Optimizations:
            - shard_number (6): Splits data into 6 pieces. Ready for future horizontal scaling.
            - on_disk (True): Raw vectors are stored on disk, keeping RAM free for fast indexing.
            - replication_factor (1): Standard for single nodes; increase for clusters.
        """
        await self._ensure_connected()
        try:
            if do_reset:
                await self.delete_collection(collection_name)
            
            if not await self.is_collection_exists(collection_name):
                await self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_size,
                        distance=self.distance_metric,
                        on_disk=True  # Saves RAM for massive datasets
                    ),
                    shard_number=6,    # Ready for horizontal multi-server distribution
                    replication_factor=1 
                )
                self.logger.info(f"Collection '{collection_name}' created successfully.")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            return False

    async def insert_one(self, collection_name: str, text: str, vector: list, 
                         metadata: dict = None, record_id: str = None):
        """
        Inserts a single vector record. 
        Uses 'upload_records' to keep structure consistent with tutorial images.
        """
        await self._ensure_connected()
        if not await self.is_collection_exists(collection_name):
            self.logger.error(f"Cannot insert: Collection '{collection_name}' not found.")
            return False

        try:
            await self.client.upload_records(
                collection_name=collection_name,
                records=[
                    models.Record(
                        id=record_id or str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": text,        # Stored for RAG retrieval
                            "metadata": metadata # Extra info (page number, source, etc.)
                        }
                    )
                ]
            )
            return True
        except Exception as e:
            self.logger.error(f"Error in insert_one: {e}")
            return False

    async def insert_many(self, collection_name: str, texts: list, vectors: list, 
                            metadatas: list = None, record_ids: list = None, batch_size: int = 50):
            """
            High-throughput batch insertion using native 'upsert' coroutines.
            Optimized for large scale by allowing true parallel I/O.
            """
            await self._ensure_connected()
            
            # 1. Validation
            if not await self.is_collection_exists(collection_name):
                self.logger.error(f"Collection '{collection_name}' not found.")
                return False
            
            count = len(texts)
            metadatas = metadatas or [{}] * count
            # Ensure we use UUIDs or standard IDs for Qdrant
            record_ids = record_ids or [str(uuid.uuid4()) for _ in range(count)]

            # 2. Parallel Batching
            tasks = []
            for i in range(0, count, batch_size):
                batch_points = [
                    models.PointStruct(
                        id=p_id,
                        vector=vec,
                        payload={"text": txt, "metadata": meta or {}}
                    )
                    for txt, vec, meta, p_id in zip(
                        texts[i:i + batch_size], 
                        vectors[i:i + batch_size], 
                        metadatas[i:i + batch_size], 
                        record_ids[i:i + batch_size]
                    )
                ]
                
                # FIX: Use 'upsert' instead of 'upload_points'.
                # 'upsert' returns a coroutine that asyncio.gather can manage.
                tasks.append(
                    self.client.upsert(
                        collection_name=collection_name,
                        points=batch_points,
                        wait=True 
                    )
                )

            # 3. Concurrent Execution
            # This will now work because 'tasks' contains valid coroutines
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 4. Error Checking
            successful_batches = 0
            for idx, res in enumerate(results):
                if isinstance(res, Exception):
                    self.logger.error(f"Batch {idx} failed: {res}")
                else:
                    successful_batches += 1
            
            return successful_batches == len(tasks)
    
    async def search_by_vector(self, collection_name: str, vector: list, limit: int = 5) -> List[RettrievedDocument]:
        """Optimized search with explicit mapping and error handling."""
        await self._ensure_connected()
        try:
            response = await self.client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=limit
            )
            
            # response.points is a list of ScoredPoint
            return [
                RettrievedDocument(
                    text=hit.payload.get("text", ""),
                    score=hit.score
                )
                for hit in response.points
            ]
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []