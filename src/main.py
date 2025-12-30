from fastapi import FastAPI

from routes import base
from routes import data

app = FastAPI()

app.include_router(base.baseRouter)
app.include_router(data.data_router)