import asyncio
import logging
import os
from pymilvus import MilvusClient
from ..VectorDBInterfaceAsync import VectorDBInterfaceAsync
from ..VectorDBEnums import DistanceMetricEnums
from models.db_schemas import RetrievedDocument
from typing import List

class MilvusAsyncProvider(VectorDBInterfaceAsync):
    """
    MilvusAsyncProvider: Asynchronous implementation of the Milvus vector database provider.
    
    This provider wraps synchronous MilvusClient calls using asyncio.to_thread()
    to provide async/await interface compatibility. Uses MilvusClient for local file storage.
    """
    
    def __init__(self, db_path: str,
                 distance_metric: str = DistanceMetricEnums.COSINE.value):
        """
        Initialize the async Milvus provider.
        
        Args:
            db_path (str): Full path to the Milvus database file (e.g., "milvus_demo.db")
            distance_metric (str): Distance metric to use (cosine, euclidean, or dot)
        """
        self.db_path = db_path
        self.client = None
        self.logger = logging.getLogger(__name__)
        
        # Map distance metrics from enum values to Milvus metric types
        metric_map = {
            DistanceMetricEnums.COSINE.value: "COSINE",
            DistanceMetricEnums.EUCLIDEAN.value: "L2",  # Milvus uses L2 for Euclidean
            DistanceMetricEnums.DOT.value: "IP"  # Inner Product for dot product
        }
        
        if distance_metric not in metric_map:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        
        self.distance_metric = metric_map[distance_metric]
    
    async def _ensure_connected(self):
        """Internal helper to ensure client is connected before operations."""
        if self.client is None:
            raise RuntimeError("Client not connected. Call connect() first.")
    
    async def connect(self):
        """
        Establish an asynchronous connection to the Milvus database.
        Creates the client with the specified database file path.
        """
        def _connect_sync():
            # Ensure the directory exists
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir, exist_ok=True)
            
            return MilvusClient(self.db_path)
        
        self.client = await asyncio.to_thread(_connect_sync)
        self.logger.info(f"Connected to Milvus (async) at: {self.db_path}")
    
    async def disconnect(self):
        """Close the connection to the Milvus database."""
        if self.client:
            self.client = None
        self.logger.info("Disconnected from Milvus")
    
    async def is_collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in Milvus.
        
        Args:
            collection_name (str): The name of the collection to check
            
        Returns:
            bool: True if collection exists, False otherwise
        """
        await self._ensure_connected()
        try:
            return await asyncio.to_thread(
                self.client.has_collection,
                collection_name=collection_name
            )
        except Exception as e:
            self.logger.error(f"Error checking if collection exists: {e}")
            return False
    
    async def list_all_collections(self) -> list:
        """
        List all collections in the Milvus database.
        
        Returns:
            list: A list of collection names (strings)
        """
        await self._ensure_connected()
        try:
            collections = await asyncio.to_thread(self.client.list_collections)
            return list(collections) if collections else []
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []
    
    async def get_collection_info(self, collection_name: str):
        """
        Retrieve information about a collection.
        
        Args:
            collection_name (str): The name of the collection
            
        Returns:
            dict: Collection metadata including size, embedding dimension, etc.
        """
        await self._ensure_connected()
        try:
            exists = await self.is_collection_exists(collection_name)
            if not exists:
                return None
            
            # MilvusClient doesn't have a direct describe_collection method
            # We'll return a basic info dict
            return {
                "collection_name": collection_name,
                "exists": True
            }
        except Exception as e:
            self.logger.error(f"Error retrieving collection info: {e}")
            return None
    
    async def delete_collection(self, collection_name: str):
        """
        Delete a collection from Milvus.
        
        Args:
            collection_name (str): The name of the collection to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        await self._ensure_connected()
        try:
            if await self.is_collection_exists(collection_name):
                await asyncio.to_thread(
                    self.client.drop_collection,
                    collection_name=collection_name
                )
                self.logger.info(f"Successfully deleted collection: {collection_name}")
                return True
            
            self.logger.warning(f"Attempted to delete non-existent collection: {collection_name}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False
    
    async def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False):
        """
        Create a new collection in Milvus.
        
        Args:
            collection_name (str): The name of the collection to create
            embedding_size (int): The dimensionality of the vectors to store
            do_reset (bool): If True, reset the collection if it already exists (default: False)
            
        Returns:
            bool: True if collection was created, False otherwise
        """
        await self._ensure_connected()
        try:
            if do_reset:
                await self.delete_collection(collection_name)
            
            if not await self.is_collection_exists(collection_name):
                # MilvusClient.create_collection uses dimension parameter
                await asyncio.to_thread(
                    self.client.create_collection,
                    collection_name=collection_name,
                    dimension=embedding_size,
                    metric_type=self.distance_metric
                )
                self.logger.info(f"Collection '{collection_name}' created with dimension {embedding_size} and metric {self.distance_metric}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            return False
    
    async def insert_one(self, collection_name: str, text: str, vector: list, 
                         metadata: dict = None, record_id: str = None):
        """
        Insert a single vector record into a collection.
        
        Args:
            collection_name (str): The target collection name
            text (str): The original text content associated with the vector
            vector (list): The embedding vector (list of floats)
            metadata (dict, optional): Additional metadata to store with the vector
            record_id (str, optional): Unique identifier for the record
            
        Returns:
            bool: True if insertion was successful, False otherwise
        """
        await self._ensure_connected()
        # Validation: Check if collection exists before attempting insert
        if not await self.is_collection_exists(collection_name):
            self.logger.error(f"Cannot insert: Collection '{collection_name}' does not exist.")
            return False
        
        try:
            # Format data as Milvus expects: list of dictionaries
            data = [{
                "id": record_id if record_id is not None else 0,  # Milvus expects numeric or string IDs
                "vector": vector,
                "text": text,
                "metadata": metadata if metadata is not None else {}
            }]
            
            await asyncio.to_thread(
                self.client.insert,
                collection_name=collection_name,
                data=data
            )
            self.logger.info(f"Successfully inserted record into '{collection_name}'")
            return True
        except Exception as e:
            self.logger.error(f"An error occurred while inserting into '{collection_name}': {e}")
            return False
    
    async def insert_many(self, collection_name: str, texts: list, vectors: list, 
                          metadatas: list = None, record_ids: list = None, batch_size: int = 50):
        """
        Insert multiple vector records into a collection in batches.
        
        Args:
            collection_name (str): The target collection name
            texts (list): List of original text contents
            vectors (list): List of embedding vectors (each vector is a list of floats)
            metadatas (list, optional): List of metadata dictionaries (one per vector)
            record_ids (list, optional): List of unique identifiers (auto-generated if not provided)
            batch_size (int): Number of records to insert per batch (default: 50)
            
        Returns:
            bool: True if the process completes, even if some batches failed (errors are logged)
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
        
        # Create batches
        tasks = []
        for i in range(0, len(texts), batch_size):
            batch_end = i + batch_size
            
            # Slice the input lists to get the current batch
            b_texts = texts[i:batch_end]
            b_vectors = vectors[i:batch_end]
            b_metadata = metadatas[i:batch_end]
            b_ids = record_ids[i:batch_end]
            
            # Format data as Milvus expects: list of dictionaries
            batch_data = [
                {
                    "id": r_id if r_id is not None else i + idx,  # Use index if no ID provided
                    "vector": vec,
                    "text": txt,
                    "metadata": meta if meta is not None else {}
                }
                for idx, (txt, vec, meta, r_id) in enumerate(zip(b_texts, b_vectors, b_metadata, b_ids))
            ]
            
            # Queue the insert task for this batch
            tasks.append(
                asyncio.to_thread(
                    self.client.insert,
                    collection_name=collection_name,
                    data=batch_data
                )
            )
        
        # Execute all batch insert tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        successful_batches = 0
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                batch_start = idx * batch_size
                self.logger.error(f"Batch processing failed at batch index {idx} (starting at record {batch_start}): {res}")
            else:
                successful_batches += 1
        
        total_batches = len(tasks)
        if successful_batches == 0:
            self.logger.error(f"All {total_batches} batches failed for collection '{collection_name}'")
            return False
        
        if successful_batches < total_batches:
            self.logger.warning(f"Only {successful_batches}/{total_batches} batches succeeded for collection '{collection_name}'")
        
        return True
    
    async def search_by_vector(self, collection_name: str,
                                vector: list, limit: int = 5) -> List[RetrievedDocument]:
        """
        Asynchronously searches for similar vectors in a Milvus collection.

        This method wraps the synchronous Milvus search call in a thread to maintain 
        async compatibility. It retrieves the associated text and maps the output 
        to the project's standardized 'RetrievedDocument' schema.

        Args:
            collection_name (str): The name of the collection to search in.
            vector (list): The query embedding vector (list of floats).
            limit (int, optional): Maximum number of results to return. Defaults to 5.

        Returns:
            list[RetrievedDocument]: A list of retrieved documents with their text 
                and similarity/distance scores. Returns an empty list [] if no 
                results are found or if an error occurs.
        """
        await self._ensure_connected()
        try:
            if not await self.is_collection_exists(collection_name):
                self.logger.error(f"Cannot search: Collection '{collection_name}' does not exist.")
                return []
            
            # 1. Perform the search in a separate thread to avoid blocking
            # Milvus returns: [[{'id': 1, 'distance': 0.9, 'entity': {...}}, ...]]
            results = await asyncio.to_thread(
                self.client.search,
                collection_name=collection_name,
                data=[vector], 
                limit=limit,
                output_fields=["text", "metadata"]
            )
            
            if not results or len(results) == 0:
                return []

            # 2. Map Milvus 'hits' to the RetrievedDocument schema
            # results[0] contains hits for our single query vector
            return [
                RetrievedDocument(
                    text=hit.get("entity", {}).get("text", ""),
                    score=hit.get("distance", 0.0)
                )
                for hit in results[0]
            ]
            
        except Exception as e:
            self.logger.error(f"Error searching in collection '{collection_name}': {e}")
            return []