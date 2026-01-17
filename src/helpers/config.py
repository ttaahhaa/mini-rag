import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# 1. Manually find the project root
# This file is in: /home/taha/mini-rag/src/helpers/config.py
current_file_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
env_path = os.path.join(project_root, ".env")

print(f"\n--- CONFIG DIAGNOSTIC ---")
print(f"Looking for .env at: {env_path}")
if os.path.exists(env_path):
    print("SUCCESS: .env file found!")
else:
    print("ERROR: .env file NOT FOUND at this path.")
print(f"--------------------------\n")

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
    GENERATION_BACKEND: str = None
    EMBEDDING_BACKEND: str = None

    GENERATION_MODEL_ID: str = None
    EMBEDDING_MODEL_ID: str = None
    EMBEDDING_MODEL_SIZE: int = None

    DEFAULT_INPUT_MAX_TOKENS: int = 1024
    DEFUALT_OUTPUT_MAX_TOKENS: int = 200
    DEFAULT_GENERATION_TEMPERATURE: float = 0.1

    # OpenAI Configurations
    OPENAI_API_KEY: str = None
    OPENAI_API_URL: str = None

    # Cohere Configurations
    COHERE_API_KEY: str = None

    model_config = SettingsConfigDict(
        env_file=env_path,
        env_file_encoding='utf-8',
        extra="ignore"
    )

    # VectorDB Configuration
    VECTOR_DB_BACKEND: str
    VECTOR_DB_NAME: str  # Database name (used to construct file paths)
    VECTOR_DB_DISTANCE_METRIC: str = None
    
def get_settings():
    return Settings()