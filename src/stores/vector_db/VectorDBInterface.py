from abc import ABC, abstractmethod

class VectorDBInterface(ABC):
    """
    Abstract interface for vector database operations.
    
    This interface defines the contract for all vector database implementations
    (e.g., Qdrant, Pinecone, Milvus). Concrete implementations must provide
    methods for connecting, managing collections, and storing/retrieving vectors.
    """
    @abstractmethod
    def connect(self):
        """
        Establish a connection to the vector database.
        
        This method should handle all necessary initialization and authentication
        required to connect to the database backend.
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """
        Close the connection to the vector database.
        
        This method should clean up any resources and gracefully close the connection.
        """
        pass
    
    @abstractmethod
    def is_collection_exists(self, collection_name):
        """
        Check if a collection exists in the vector database.
        
        Args:
            collection_name (str): The name of the collection to check
            
        Returns:
            bool: True if collection exists, False otherwise
        """
        pass

    @abstractmethod
    def create_collection(self, collection_name: str, embedding_size: int,
                           do_reset: bool = False):
        """
        Create a new collection in the vector database.
        
        Args:
            collection_name (str): The name of the collection to create
            embedding_size (int): The dimensionality of the vectors to store
            do_reset (bool): If True, reset the collection if it already exists (default: False)
        """
        pass
    @abstractmethod
    def insert_one(self, collection_name: str, text: str,
                   vector: list, metadata: dict,
                   record_id: str = None):
        """
        Insert a single vector record into a collection.
        
        Args:
            collection_name (str): The target collection name
            text (str): The original text content associated with the vector
            vector (list): The embedding vector (list of floats)
            metadata (dict): Additional metadata to store with the vector
            record_id (str, optional): Unique identifier for the record (auto-generated if not provided)
        """
        pass
    
    @abstractmethod
    def insert_many(self, collection_name: str, texts: list,
                    vectors: list, metadatas: list,
                    record_ids: list = None, batch_size: int = 5):
        """
        Insert multiple vector records into a collection in batches.
        
        Args:
            collection_name (str): The target collection name
            texts (list): List of original text contents
            vectors (list): List of embedding vectors (each vector is a list of floats)
            metadatas (list): List of metadata dictionaries (one per vector)
            record_ids (list, optional): List of unique identifiers (auto-generated if not provided)
            batch_size (int): Number of records to insert per batch (default: 5)
        """
        pass    

    @abstractmethod
    def get_collection_info(self, collection_name):
        """
        Retrieve information about a collection.
        
        Args:
            collection_name (str): The name of the collection
            
        Returns:
            dict: Collection metadata including size, embedding dimension, etc.
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_name):
        """
        Delete a collection from the vector database.
        
        Args:
            collection_name (str): The name of the collection to delete
        """
        pass