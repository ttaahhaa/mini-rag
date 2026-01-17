from fastapi import FastAPI
from contextlib import asynccontextmanager # 1. Import this
from routes import base, data, nlp
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings
from stores.llm.LLMProvidorFactory import LLMProviderFactory
from stores.vector_db.VectorDBProviderFactory import VectorDBProviderFactory

# 2. Define the lifespan logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    settings = get_settings()
    
    # Initialize the client
    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]
    
    print("SUCCESS: MongoDB connection established.")

    llm_provider_factory = LLMProviderFactory(config=settings)
    vector_DB_provider_factory = VectorDBProviderFactory(config=settings)
    # Initialize generation client
    app.generation_client = llm_provider_factory.create(settings.GENERATION_BACKEND)
    app.generation_client.set_generation_model(model_id=settings.GENERATION_MODEL_ID)

    # Initialize embedding client
    app.embedding_client = llm_provider_factory.create(settings.EMBEDDING_BACKEND)
    app.embedding_client.set_embedding_model(
        model_id=settings.EMBEDDING_MODEL_ID,
        embedding_size=settings.EMBEDDING_MODEL_SIZE
    )

    # initialize db vector
    app.vectordb_client = vector_DB_provider_factory.create(settings.VECTOR_DB_BACKEND)
    app.vectordb_client.connect()

    yield  # 3. This is where the app actually "runs"

   # --- Shutdown ---
    app.mongo_conn.close()
    print("SUCCESS: MongoDB connection closed.")

    if hasattr(app, 'vectordb_client'):
        app.vectordb_client.disconnect()
        print("SUCCESS: VectorDB connection closed.")

# 4. Pass the lifespan function to the FastAPI instance
app = FastAPI(lifespan=lifespan)

app.include_router(base.baseRouter)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)