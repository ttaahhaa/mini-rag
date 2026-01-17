import uuid
import logging
import asyncio
from qdrant_client import AsyncQdrantClient, models
from ..VectorDBInterfaceAsync import VectorDBInterfaceAsync
from ..VectorDBEnums import DistanceMetricEnums

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
        """
        Establishes an asynchronous connection to Qdrant.
        - If 'url' is provided: Connects to a remote server or Docker cluster.
        - If 'url' is None: Uses local file-based 'Embedded' mode at 'db_path'.
        """
        if self.url:
            self.client = AsyncQdrantClient(url=self.url, api_key=self.api_key)
        else:
            self.client = AsyncQdrantClient(path=self.db_path)
        self.logger.info("Successfully connected to Async Qdrant client.")

    async def disconnect(self):
        """Gracefully closes the connection by nullifying the client."""
        if self.client:
            # AsyncQdrantClient should be closed properly if it has a close method
            if hasattr(self.client, 'close'):
                await self.client.close()
        self.client = None
        self.logger.info("Disconnected from Qdrant.")

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
        High-throughput batch insertion using parallelized async tasks.

        Optimization:
            - Uses 'asyncio.gather' to send multiple batches to Qdrant concurrently.
            - Uses 'zip' for cleaner, more efficient record mapping.
        """
        await self._ensure_connected()
        
        # Validation: Check if collection exists before attempting insert
        if not await self.is_collection_exists(collection_name):
            self.logger.error(f"Cannot insert: Collection '{collection_name}' does not exist.")
            return False
        
        if metadatas is None: 
            metadatas = [None] * len(texts)
        if record_ids is None: 
            record_ids = [None] * len(texts)

        # Validate input lengths match
        if not (len(texts) == len(vectors) == len(metadatas) == len(record_ids)):
            self.logger.error("Input lists must have the same length")
            return False

        tasks = []
        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size
            b_texts = texts[i:batch_end]
            b_vectors = vectors[i:batch_end]
            b_metadata = metadatas[i:batch_end]
            b_ids = record_ids[i:batch_end]

            # Build the list of records for this specific batch
            batch_records = [
                models.Record(
                    id=r_id or str(uuid.uuid4()),
                    vector=vec,
                    payload={"text": txt, "metadata": meta}
                )
                for txt, vec, meta, r_id in zip(b_texts, b_vectors, b_metadata, b_ids)
            ]
            
            # Queue the upload task for this batch
            tasks.append(self.client.upload_records(collection_name, batch_records))

        # CONCURRENCY POWER: Execute all batch upload tasks at the same time
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_batches = 0
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                self.logger.error(f"Batch processing failed at batch index {idx}: {res}")
            else:
                successful_batches += 1
        
        total_batches = len(tasks)
        if successful_batches == 0:
            self.logger.error(f"All {total_batches} batches failed for collection '{collection_name}'")
            return False
        
        if successful_batches < total_batches:
            self.logger.warning(f"Only {successful_batches}/{total_batches} batches succeeded for collection '{collection_name}'")
        
        return True

    async def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):
        """
        Performs an asynchronous semantic search.
        Returns: Top 'limit' most similar points found.
        """
        await self._ensure_connected()
        try:
            if not await self.is_collection_exists(collection_name):
                self.logger.error(f"Cannot search: Collection '{collection_name}' does not exist.")
                return []
            
            return await self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=limit
            )
        except Exception as e:
            self.logger.error(f"Search error in collection '{collection_name}': {e}")
            return []