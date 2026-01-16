from .providers.QDrantProvider import QDrantProvider
from .providers.QDrantAsyncProvider import QDrantAsyncProvider
from .VectorDBEnums import LLMEnums # Using the Enum class you updated

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

    def create(self, provider: str):
        """
        Returns the appropriate Vector DB provider instance.

        Args:
            provider (str): The value from LLMEnums (e.g., "QDRANT" or "AsyncQDRANT").
        """
        # Logic for Synchronous Qdrant (Standard Tutorial Version)
        if provider == LLMEnums.QDRANT.value:
            return QDrantProvider(
                db_path=self.config.VECTOR_DB_PATH,
                distance_metric=self.config.VECTOR_DB_DISTANCE_METHOD
            )

        # Logic for Asynchronous Qdrant (Optimized v2 Version)
        elif provider == LLMEnums.AsyncQDRANT.value:
            return QDrantAsyncProvider(
                db_path=self.config.VECTOR_DB_PATH,
                # Note: v2 also supports remote URLs if added to your config
                url=getattr(self.config, 'VECTOR_DB_URL', None), 
                api_key=getattr(self.config, 'VECTOR_DB_API_KEY', None),
                distance_metric=self.config.VECTOR_DB_DISTANCE_METHOD
            )
        
        # Log error if an unknown provider is requested
        raise ValueError(f"Unknown Vector DB provider requested: {provider}")