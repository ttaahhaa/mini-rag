from ..VectorDBInterface import VectorDBInterface
from pymilvus import MilvusClient
import logging
import os
from ..VectorDBEnums import DistanceMetricEnums

class MilvusProvider(VectorDBInterface):
    """
    MilvusProvider: Synchronous implementation of the Milvus vector database provider.
    
    This provider uses MilvusClient for local file storage, following the same
    interface contract as other vector database providers in the system.
    """
    
    def __init__(self, db_path: str,
                 distance_metric: str = DistanceMetricEnums.COSINE.value):
        """
        Initialize the Milvus provider.
        
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
    
    def _ensure_connected(self):
        """Internal helper to ensure client is connected before operations."""
        if self.client is None:
            raise RuntimeError("Client not connected. Call connect() first.")
    
    def connect(self):
        """
        Establish a connection to the Milvus database.
        Creates the client with the specified database file path.
        """
        # Ensure the directory exists
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        
        self.client = MilvusClient(self.db_path)
        self.logger.info(f"Connected to Milvus at: {self.db_path}")
    
    def disconnect(self):
        """Close the connection to the Milvus database."""
        if self.client:
            # MilvusClient doesn't have an explicit close method in Lite mode
            # Setting to None is sufficient
            self.client = None
        self.logger.info("Disconnected from Milvus")
    
    def is_collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists in Milvus.
        
        Args:
            collection_name (str): The name of the collection to check
            
        Returns:
            bool: True if collection exists, False otherwise
        """
        self._ensure_connected()
        try:
            return self.client.has_collection(collection_name=collection_name)
        except Exception as e:
            self.logger.error(f"Error checking if collection exists: {e}")
            return False
    
    def list_all_collections(self) -> list:
        """
        List all collections in the Milvus database.
        
        Returns:
            list: A list of collection names (strings)
        """
        self._ensure_connected()
        try:
            collections = self.client.list_collections()
            return list(collections) if collections else []
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []
    
    def get_collection_info(self, collection_name: str):
        """
        Retrieve information about a collection.
        
        Args:
            collection_name (str): The name of the collection
            
        Returns:
            dict: Collection metadata including size, embedding dimension, etc.
        """
        self._ensure_connected()
        try:
            if not self.is_collection_exists(collection_name):
                return None
            
            # MilvusClient doesn't have a direct describe_collection method
            # We'll return a basic info dict
            # Note: For full schema info, would need to use Collection.describe() from full Milvus client
            return {
                "collection_name": collection_name,
                "exists": True
            }
        except Exception as e:
            self.logger.error(f"Error retrieving collection info: {e}")
            return None
    
    def delete_collection(self, collection_name: str):
        """
        Delete a collection from Milvus.
        
        Args:
            collection_name (str): The name of the collection to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        self._ensure_connected()
        try:
            if self.is_collection_exists(collection_name):
                self.client.drop_collection(collection_name=collection_name)
                self.logger.info(f"Successfully deleted collection: {collection_name}")
                return True
            
            self.logger.warning(f"Attempted to delete non-existent collection: {collection_name}")
            return False
        except Exception as e:
            self.logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False
    
    def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False):
        """
        Create a new collection in Milvus.
        
        Args:
            collection_name (str): The name of the collection to create
            embedding_size (int): The dimensionality of the vectors to store
            do_reset (bool): If True, reset the collection if it already exists (default: False)
            
        Returns:
            bool: True if collection was created, False otherwise
        """
        self._ensure_connected()
        try:
            if do_reset:
                self.delete_collection(collection_name)
            
            if not self.is_collection_exists(collection_name):
                # MilvusClient.create_collection uses dimension parameter
                # Note: metric_type defaults to COSINE, but we'll set it explicitly
                self.client.create_collection(
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
    
    def insert_one(self, collection_name: str, text: str, vector: list, 
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
        self._ensure_connected()
        # Validation: Check if collection exists before attempting insert
        if not self.is_collection_exists(collection_name):
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
            
            result = self.client.insert(collection_name=collection_name, data=data)
            self.logger.info(f"Successfully inserted record into '{collection_name}'")
            return True
        except Exception as e:
            self.logger.error(f"An error occurred while inserting into '{collection_name}': {e}")
            return False
    
    def insert_many(self, collection_name: str, texts: list,
                    vectors: list, metadatas: list = None,
                    record_ids: list = None, batch_size: int = 50):
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
        self._ensure_connected()
        # Validation: Check if collection exists before attempting insert
        if not self.is_collection_exists(collection_name):
            self.logger.error(f"Cannot insert: Collection '{collection_name}' does not exist.")
            return False
        
        # Initialize metadatas and record_ids as lists of None if not provided
        if metadatas is None:
            metadatas = [None] * len(texts)
        
        if record_ids is None:
            record_ids = [None] * len(texts)
        
        # Validate input lengths match
        if not (len(texts) == len(vectors) == len(metadatas) == len(record_ids)):
            self.logger.error("Input lists must have the same length")
            return False
        
        successful_batches = 0
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        # Process data in chunks to optimize memory and network usage
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
            
            try:
                # Send the current batch to the database
                self.client.insert(collection_name=collection_name, data=batch_data)
                successful_batches += 1
            except Exception as e:
                # Log the error for this specific batch and continue with the next
                self.logger.error(f"Error while inserting batch starting at index {i}: {e}")
        
        if successful_batches == 0:
            self.logger.error(f"All {total_batches} batches failed for collection '{collection_name}'")
            return False
        
        if successful_batches < total_batches:
            self.logger.warning(f"Only {successful_batches}/{total_batches} batches succeeded for collection '{collection_name}'")
        
        return True
    
    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):
        """
        Search for similar vectors in a collection.
        
        Args:
            collection_name (str): The name of the collection to search in
            vector (list): The query vector (list of floats)
            limit (int): Maximum number of results to return (default: 5)
            
        Returns:
            list: List of search results with scores and payloads, normalized to match Qdrant format
        """
        self._ensure_connected()
        try:
            if not self.is_collection_exists(collection_name):
                self.logger.error(f"Cannot search: Collection '{collection_name}' does not exist.")
                return []
            
            # Milvus search returns results in format:
            # [{"id": x, "distance": y, "entity": {"text": "...", "metadata": {...}}}]
            results = self.client.search(
                collection_name=collection_name,
                data=[vector],  # Milvus expects a list of vectors
                limit=limit,
                output_fields=["text", "metadata"]
            )
            
            # Normalize results to match Qdrant format
            # Qdrant returns: [ScoredPoint(id=..., score=..., payload={...})]
            # We'll convert to a similar dict format for consistency
            normalized_results = []
            if results and len(results) > 0:
                # results is a list where each element corresponds to a query vector
                # Since we only pass one vector, results[0] contains the matches
                for hit in results[0]:
                    normalized_results.append({
                        "id": hit.get("id"),
                        "score": hit.get("distance"),
                        "payload": {
                            "text": hit.get("entity", {}).get("text", ""),
                            "metadata": hit.get("entity", {}).get("metadata", {})
                        }
                    })
            
            return normalized_results
        except Exception as e:
            self.logger.error(f"Error searching in collection '{collection_name}': {e}")
            return []
