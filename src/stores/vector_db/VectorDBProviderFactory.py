import os
import logging
from .providers.QDrantProvider import QDrantProvider
from .providers.QDrantAsyncProvider import QDrantAsyncProvider
from .providers.MilvusProvider import MilvusProvider
from .providers.MilvusAsyncProvider import MilvusAsyncProvider
from helpers.config import VectorDBEnums 
from controllers.BaseAsyncController import BaseAsyncController

class VectorDBProviderFactory:
    """
    Factory class to instantiate Vector Database providers.
    Designed for scalability, supporting both Sync and Async backends.
    """

    def __init__(self, config):
        """
        Initializes the factory with the project settings.
        """
        self.config = config
        self.base_controller = BaseAsyncController()
        self.logger = logging.getLogger(__name__)

    async def create(self, provider: str):
        """
        Returns the appropriate Vector DB provider instance.

        Args:
            provider (str): The value from VectorDBEnums (e.g., "AsyncQDRANT").
        """
        # Validate critical configuration
        if not self.config.VECTOR_DB_NAME:
            raise ValueError("VECTOR_DB_NAME is missing in configuration.")
        
        if not self.config.VECTOR_DB_DISTANCE_METRIC:
            raise ValueError("VECTOR_DB_DISTANCE_METRIC is missing in configuration.")

        # 1. Logic for Asynchronous Qdrant (Optimized for Scale)
        if provider == VectorDBEnums.AsyncQDRANT.value:
            # Await the directory path creation
            db_path = await self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_NAME)

            return QDrantAsyncProvider(
                db_path=db_path,
                url=self.config.VECTOR_DB_URL, 
                api_key=self.config.VECTOR_DB_API_KEY,
                distance_metric=self.config.VECTOR_DB_DISTANCE_METRIC
            )

        # 2. Logic for Synchronous Qdrant (Fallback)
        elif provider == VectorDBEnums.QDRANT.value:
            db_path = await self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_NAME)

            return QDrantProvider(
                db_path=db_path,
                distance_metric=self.config.VECTOR_DB_DISTANCE_METRIC
            )

        # 3. Logic for Asynchronous Milvus (Optimized for Scale)
        elif provider == VectorDBEnums.AsyncMILVUS.value:
            db_path = await self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_NAME)
            milvus_db_file = os.path.join(db_path, f"milvus_{self.config.VECTOR_DB_NAME}.db")

            return MilvusAsyncProvider(
                db_path=milvus_db_file,
                distance_metric=self.config.VECTOR_DB_DISTANCE_METRIC
            )

        # 4. Logic for Synchronous Milvus
        elif provider == VectorDBEnums.MILVUS.value:
            db_path = await self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_NAME)
            milvus_db_file = os.path.join(db_path, f"milvus_{self.config.VECTOR_DB_NAME}.db")

            return MilvusProvider(
                db_path=milvus_db_file,
                distance_metric=self.config.VECTOR_DB_DISTANCE_METRIC
            )
        
        # Explicitly raise an error if the provider string is invalid
        raise ValueError(f"Unknown Vector DB provider: {provider}. Please check your .env file.")