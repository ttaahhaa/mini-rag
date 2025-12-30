from fastapi import FastAPI, APIRouter, Depends
from helpers.config import get_settings, Settings

baseRouter = APIRouter(
    prefix="/api/v1",
    tags=["api_v1"]
)


@baseRouter.get("/health")
def health(app_settings: Settings = Depends(get_settings)):
    app_settings = get_settings()
    app_name = app_settings.APP_NAME
    app_version = app_settings.APP_VERSION
    return {
        "message": "welcome to Mini-Rag",
        "APP_NAME": app_name,
        "APP_VERSION": app_version
    }