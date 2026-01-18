import os
from enum import Enum
from pydantic_settings import BaseSettings, SettingsConfigDict

# --- Enums for Type Safety ---

class LLMEnums(Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    ANTHROPIC = "anthropic"

class VectorDBEnums(Enum):
    QDRANT = "QDRANT"
    AsyncQDRANT = "AsyncQDRANT"
    MILVUS = "MILVUS"
    AsyncMILVUS = "AsyncMILVUS"

class DocumentTypeEnum(Enum):
    DOCUMENT = "document"
    QUERY = "query"

# --- Path Resolution ---
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
env_path = os.path.join(project_root, ".env")

# --- Settings Class ---
class Settings(BaseSettings):
    # App Settings
    APP_NAME: str
    APP_VERSION: str

    # File Settings
    FILE_ALLOWED_TYPES: list
    FILE_MAX_SIZE: int
    FILE_DEFAULT_CHUNK_SIZE: int
    
    # Database Settings
    MONGODB_URL: str
    MONGODB_DATABASE: str

    # LLM Configurations
    GENERATION_BACKEND: str
    EMBEDDING_BACKEND: str
    GENERATION_MODEL_ID: str
    EMBEDDING_MODEL_ID: str
    EMBEDDING_MODEL_SIZE: int

    DEFAULT_INPUT_MAX_TOKENS: int = 1024
    DEFUALT_OUTPUT_MAX_TOKENS: int = 200
    DEFAULT_GENERATION_TEMPERATURE: float = 0.1

    # Provider Keys
    OPENAI_API_KEY: str = None
    OPENAI_API_URL: str = None
    COHERE_API_KEY: str = None

    # VectorDB Configuration (UPDATED for Scale)
    VECTOR_DB_BACKEND: str
    VECTOR_DB_NAME: str
    VECTOR_DB_DISTANCE_METRIC: str
    # Add these for remote clusters/Docker
    VECTOR_DB_URL: str = None
    VECTOR_DB_API_KEY: str = None

    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding='utf-8',
        extra="ignore"
    )

def get_settings():
    return Settings()