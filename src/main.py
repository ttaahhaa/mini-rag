from fastapi import FastAPI
from contextlib import asynccontextmanager
from routes import base, data, nlp
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from stores.llm.LLMProvidorFactory import LLMProviderFactory
from stores.vector_db.VectorDBProviderFactory import VectorDBProviderFactory
from stores.templates import TemplateParser
from models.enums.TemplatesEnum import TemplateLanguagesEnums

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    settings = get_settings()
    
    # 1. Initialize MongoDB
    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]
    print("SUCCESS: MongoDB connection established.")

    llm_provider_factory = LLMProviderFactory(config=settings)
    vector_DB_provider_factory = VectorDBProviderFactory(config=settings)

    # 2. Initialize generation client
    app.generation_client = llm_provider_factory.create(settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    # 3. Initialize embedding client
    app.embedding_client = llm_provider_factory.create(settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE
    )

    # 4. Initialize Vector DB (FIXED: Added await for async create)
    # Since your factory's 'create' is now 'async def', you must await it.
    app.vectordb_client = await vector_DB_provider_factory.create(settings.VECTOR_DB_BACKEND)
    if app.vectordb_client is None:
        raise RuntimeError(
            f"VectorDB Client could not be initialized. "
            f"Check if '{settings.VECTOR_DB_BACKEND}' is correctly handled in the Factory."
        )    
    await app.vectordb_client.connect()
    print(f"SUCCESS: VectorDB ({settings.VECTOR_DB_BACKEND}) connection established.")

    app.template_parser = TemplateParser(default_language=settings.DEFAULT_LANG,
                                         language=settings.PRIMARY_LANG)

    yield  # Application runs here

    # --- Shutdown ---
    app.mongo_conn.close()
    print("SUCCESS: MongoDB connection closed.")

    if hasattr(app, 'vectordb_client'):
        # FIXED: Added await for the disconnection call
        await app.vectordb_client.disconnect()
        print("SUCCESS: VectorDB connection closed.")

# Initialize FastAPI with the corrected lifespan
app = FastAPI(lifespan=lifespan)

app.include_router(base.baseRouter)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)