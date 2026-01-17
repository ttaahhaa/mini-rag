from ..VectorDBInterface import VectorDBInterface
from qdrant_client import QdrantClient, models
import logging
from ..VectorDBEnums import DistanceMetricEnums
import uuid

class QDrantProvider(VectorDBInterface):
    def __init__(self, db_path: str,
                 distance_metric: str = DistanceMetricEnums.COSINE.value):
        self.db_path = db_path
        self.distance_metric = None
        self.client = None
        self.logger = logging.getLogger(__name__)
        
        if distance_metric == DistanceMetricEnums.COSINE.value:
            self.distance_metric = models.Distance.COSINE
        elif distance_metric == DistanceMetricEnums.EUCLIDEAN.value:
            self.distance_metric = models.Distance.EUCLID
        elif distance_metric == DistanceMetricEnums.DOT.value:
            self.distance_metric = models.Distance.DOT
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

    def _ensure_connected(self):
        """Internal helper to ensure client is connected before operations."""
        if self.client is None:
            raise RuntimeError("Client not connected. Call connect() first.")

    def connect(self):
        self.client = QdrantClient(path=self.db_path)
        self.logger.info(f"Connected to Qdrant at: {self.db_path}")

    def disconnect(self):
        self.client = None
        self.logger.info("Disconnected from Qdrant")

    def is_collection_exists(self, collection_name: str) -> bool:
        self._ensure_connected()
        try:
            # Note: client.collection_exists returns a boolean directly
            return self.client.collection_exists(collection_name)
        except Exception as e:
            self.logger.error(f"Error checking if collection exists: {e}")
            return False

    def list_all_collections(self) -> list:
        """
        Retrieves a list of all existing collection names in the database.
        
        Returns:
            list: A list of collection names (strings)
        """
        self._ensure_connected()
        try:
            collections_response = self.client.get_collections()
            return [col.name for col in collections_response.collections]
        except Exception as e:
            self.logger.error(f"Error listing collections: {e}")
            return []
    
    def get_collection_info(self, collection_name: str):
        self._ensure_connected()
        try:
            return self.client.get_collection(collection_name=collection_name)
        except Exception as e:
            self.logger.error(f"Error retrieving collection info: {e}")
            return None
        
    def delete_collection(self, collection_name: str):
        self._ensure_connected()
        try:
            if self.is_collection_exists(collection_name):
                result = self.client.delete_collection(collection_name=collection_name)
                self.logger.info(f"Successfully deleted collection: {collection_name}")
                return result
            
            self.logger.warning(f"Attempted to delete non-existent collection: {collection_name}")
            return False

        except Exception as e:
            self.logger.error(f"Failed to delete collection '{collection_name}': {e}")
            return False

    def create_collection(self, collection_name: str, embedding_size: int, do_reset: bool = False):
        self._ensure_connected()
        try:
            if do_reset:
                self.delete_collection(collection_name)
            
            if not self.is_collection_exists(collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=models.VectorParams(
                        size=embedding_size,
                        distance=self.distance_metric
                    ),
                    # Scalability: Prepare for 3+ nodes even while testing locally
                    shard_number=6, 
                    replication_factor=1
                )
                self.logger.info(f"Collection '{collection_name}' created.")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error creating collection: {e}")
            return False

    def insert_one(self, collection_name: str, text: str, vector: list, 
               metadata: dict = None, record_id: str = None):
        """
        Inserts a single record into Qdrant using the upload_records method.
        Includes try-except error handling as recommended in the tutorial.
        """
        self._ensure_connected()
    
        # 1. Validation: Check if collection exists
        if not self.is_collection_exists(collection_name):
            self.logger.error(f"Cannot insert: Collection '{collection_name}' does not exist.")
            return False

        # 2. FIX: Ensure record_id is never None (prevents Pydantic ValidationError)
        if record_id is None:
            record_id = str(uuid.uuid4())

        try:
            _ = self.client.upload_points(
                collection_name=collection_name,
                points=[ 
                    models.Record(
                        id=record_id,
                        vector=vector,
                        payload={
                            "text": text,
                            "metadata": metadata
                        }
                    )
                ]
            )
            self.logger.info(f"Successfully inserted record into '{collection_name}'")
            return True

        except Exception as e:
            self.logger.error(f"An error occurred while inserting into '{collection_name}': {e}")
            return False

    def insert_many(self, collection_name: str, texts: list,
                    vectors: list, metadatas: list = None,
                    record_ids: list = None, batch_size: int = 50):
        """
        Inserts a list of records into a Qdrant collection using batch processing.

        This function optimizes the insertion process by splitting large datasets into smaller 
        chunks (batches). This prevents memory spikes and ensures stable performance when 
        handling high volumes of data. It uses Python's `zip` function to align 
        text, vectors, and metadata efficiently.

        Args:
            collection_name (str): The name of the target Qdrant collection.
            texts (list): A list of strings representing the original content to be stored.
            vectors (list): A list of embedding vectors (list of floats) corresponding to the texts.
            metadatas (list, optional): A list of dictionaries containing extra info (e.g., source, date). 
                Defaults to None.
            record_ids (list, optional): A list of unique IDs for each record. If None, Qdrant 
                generates IDs automatically. Defaults to None.
            batch_size (int, optional): Number of records to send in each request. 
                Defaults to 50.

        Returns:
            bool: True if the process completes, even if some batches failed (errors are logged).
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

            # Use zip to pair elements and create a list of Record objects
            batch_records = [
                models.Record(
                    id=r_id,
                    vector=vec,
                    payload={
                        "text": txt,        # Stored for retrieval by the LLM later
                        "metadata": meta    # Nested metadata dictionary
                    }
                )
                for txt, vec, meta, r_id in zip(b_texts, b_vectors, b_metadata, b_ids)
            ]

            try:
                # Send the current batch to the database
                self.client.upload_points(
                    collection_name=collection_name,
                    points=batch_records,
                )
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
            list: List of search results with scores and payloads
        """
        self._ensure_connected()
        try:
            if not self.is_collection_exists(collection_name):
                self.logger.error(f"Cannot search: Collection '{collection_name}' does not exist.")
                return []
            
            # Use query_points instead of search for modern qdrant-client versions
            return self.client.query_points(
                collection_name=collection_name,
                query=vector,  # In query_points, the parameter name is 'query'
                limit=limit
            )
        except Exception as e:
            self.logger.error(f"Error searching in collection '{collection_name}': {e}")
            return []