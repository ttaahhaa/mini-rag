from .providers.QDrantProvider import QDrantProvider
from .providers.QDrantAsyncProvider import QDrantAsyncProvider
from .providers.MilvusProvider import MilvusProvider
from .providers.MilvusAsyncProvider import MilvusAsyncProvider
from .VectorDBEnums import LLMEnums 
from controllers.BaseController import BaseController
import os

class VectorDBProviderFactory:
    """
    Factory class to instantiate either synchronous or asynchronous 
    Vector Database providers based on the project configuration.
    """

    def __init__(self, config):
        """
        Initializes the factory with the global configuration object.
        """
        self.config = config
        self.base_controller = BaseController()

    def create(self, provider: str):
        """
        Returns the appropriate Vector DB provider instance.

        Args:
            provider (str): The value from LLMEnums (e.g., "QDRANT" or "AsyncQDRANT").
        """
        # Validate required config attributes
        if not hasattr(self.config, 'VECTOR_DB_NAME') or not self.config.VECTOR_DB_NAME:
            raise ValueError("VECTOR_DB_NAME must be set in configuration")
        
        if not hasattr(self.config, 'VECTOR_DB_DISTANCE_METRIC') or not self.config.VECTOR_DB_DISTANCE_METRIC:
            raise ValueError("VECTOR_DB_DISTANCE_METRIC must be set in configuration")
        
        # Logic for Synchronous Qdrant (Standard Tutorial Version)
        if provider == LLMEnums.QDRANT.value:
            db_path = self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_NAME)

            return QDrantProvider(
                db_path=db_path,
                distance_metric=self.config.VECTOR_DB_DISTANCE_METRIC
            )

        # Logic for Asynchronous Qdrant (Optimized v2 Version)
        elif provider == LLMEnums.AsyncQDRANT.value:
            db_path = self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_NAME)

            return QDrantAsyncProvider(
                db_path=db_path,
                # Note: v2 also supports remote URLs if added to your config
                url=getattr(self.config, 'VECTOR_DB_URL', None), 
                api_key=getattr(self.config, 'VECTOR_DB_API_KEY', None),
                distance_metric=self.config.VECTOR_DB_DISTANCE_METRIC
            )
        
        # Logic for Synchronous Milvus (Local file storage)
        elif provider == LLMEnums.MILVUS.value:
            db_path = self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_NAME)
            # MilvusClient expects a file path, construct it
            milvus_db_file = os.path.join(db_path, f"milvus_{self.config.VECTOR_DB_NAME}.db")

            return MilvusProvider(
                db_path=milvus_db_file,
                distance_metric=self.config.VECTOR_DB_DISTANCE_METRIC
            )

        # Logic for Asynchronous Milvus (Local file storage)
        elif provider == LLMEnums.AsyncMILVUS.value:
            db_path = self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_NAME)
            # MilvusClient expects a file path, construct it
            milvus_db_file = os.path.join(db_path, f"milvus_{self.config.VECTOR_DB_NAME}.db")

            return MilvusAsyncProvider(
                db_path=milvus_db_file,
                distance_metric=self.config.VECTOR_DB_DISTANCE_METRIC
            )
        
        # Log error if an unknown provider is requested
        raise ValueError(f"Unknown Vector DB provider requested: {provider}")