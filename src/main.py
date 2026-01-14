from fastapi import FastAPI
from contextlib import asynccontextmanager # 1. Import this
from routes import base, data 
from motor.motor_asyncio import AsyncIOMotorClient
from helpers.config import get_settings

# 2. Define the lifespan logic
@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    settings = get_settings()
    
    # Initialize the client
    app.mongo_conn = AsyncIOMotorClient(settings.MONGODB_URL)
    app.db_client = app.mongo_conn[settings.MONGODB_DATABASE]
    
    print("SUCCESS: MongoDB connection established.")

    yield  # 3. This is where the app actually "runs"

    # --- Shutdown Logic ---
    app.mongo_conn.close()
    print("SUCCESS: MongoDB connection closed.")

# 4. Pass the lifespan function to the FastAPI instance
app = FastAPI(lifespan=lifespan)

app.include_router(base.baseRouter)
app.include_router(data.data_router)