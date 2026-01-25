from fastapi import APIRouter, Depends, UploadFile, status, Request
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from .schemas import ProcessRequest

from models import ProjectModel, ChunkModel
from models import ResponseSignal
from models.AssetModel import AssetModel
from controllers.DataAsyncController import DataAsyncController
from models import AssetTypeEnum
from models.db_schemas import DataChunk, Asset
from controllers import ProcessAsyncController

# NEW: Import your db helper and type hint
from helpers.db import get_db
from sqlalchemy.ext.asyncio import AsyncSession

import aiofiles
import logging

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1_data", "data"]
)

@data_router.post("/upload/{project_id}")
async def upload_data(request: Request, project_id: int, file: UploadFile,
                      app_settings: Settings = Depends(get_settings),
                      db_session: AsyncSession = Depends(get_db)): # Added session dependency

    # UPDATE: Pass db_session instead of db_client
    project_model = await ProjectModel.create_instance(db_session=db_session)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    data_controller = DataAsyncController()
    is_valid, result_signal = data_controller.validate_uploaded_file(file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": result_signal}
        )

    file_path, file_id = await data_controller.generate_unique_filepath(
        org_file_name=file.filename,
        project_id=project_id
    )
    
    try:
        async with aiofiles.open(file_path, "wb") as f:
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e:
        logger.error(f"Error while uploading file: {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.FILE_UPLOAD_FAILED.value}
        )

    # UPDATE: Pass db_session
    asset_model = await AssetModel.create_instance(db_session=db_session)
    
    import asyncio
    file_size = await asyncio.to_thread(os.path.getsize, file_path)

    asset_record = Asset(
        asset_project_id=project.project_id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_name=file_id, 
        asset_size=file_size,
        asset_config={} 
    )
    asset_record = await asset_model.create_asset(asset=asset_record)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseSignal.FILE_UPLOAD_SUCESS.value,
            "file_id": str(asset_record.asset_id), 
        },
    )
    
@data_router.post("/process/{project_id}")
async def process_endpoint(request: Request, project_id: int, process_request: ProcessRequest,
                           db_session: AsyncSession = Depends(get_db)): # Added session dependency
    
    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size

    # UPDATE: Pass db_session to all model instances
    project_model = await ProjectModel.create_instance(db_session=db_session)
    chunk_model = await ChunkModel.create_instance(db_session=db_session)
    asset_model = await AssetModel.create_instance(db_session=db_session)
    
    project = await project_model.get_project_or_create_one(project_id=project_id)
    
    process_controller = ProcessAsyncController(project_id=project_id)
    
    project_file_ids = {}
    if process_request.file_id:
        asset_record = await asset_model.get_asset_record(
            asset_project_id=project.project_id,
            asset_name=process_request.file_id
        )
        if asset_record:
            project_file_ids = {asset_record.asset_id: asset_record.asset_name}
        else:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": ResponseSignal.FILE_NOT_FOUND_IN_PROJECT.value}
            )
    else:
        project_assets = await asset_model.get_all_project_assets(
            asset_project_id=project.project_id, 
            asset_type=AssetTypeEnum.FILE.value
        )
        project_file_ids = {asset.asset_id: asset.asset_name for asset in project_assets}
    
    if not project_file_ids:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.NO_FILES_IN_PROJECT.value}
        )
    
    if process_request.do_reset == 1:
        await chunk_model.delete_chunks_by_project_id(project_id=project.project_id)

    total_inserted = 0

    for asset_id, file_name in project_file_ids.items():
        file_content = await process_controller.get_file_content(file_id=file_name)

        if not file_content:
            logger.warning(f"Skipping file {file_name}: unable to load content.")
            continue

        file_chunks = await process_controller.process_file_content(
            file_content=file_content,
            chunk_size=chunk_size,
            chunk_overlap=overlap_size
        )

        if file_chunks:
            file_chunks_records = [
                DataChunk(
                    chunk_text=chunk.page_content,
                    chunk_metadata=chunk.metadata,
                    chunk_order=i + 1,
                    chunk_project_id=project.project_id,
                    chunk_asset_id=asset_id
                )
                for i, chunk in enumerate(file_chunks)
            ]
            
            total_inserted += await chunk_model.insert_many_chunks(chunks=file_chunks_records)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseSignal.PROCESSING_SUCESS.value,
            "number_of_chunks": total_inserted,
            "processed_files": len(project_file_ids)
        },
    )