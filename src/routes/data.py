from fastapi import FastAPI, APIRouter, Depends, UploadFile, status, Request
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController
from .schemas import ProcessRequest

from models import Asset
from models import ProjectModel, ChunkModel
from models import ResponseSignal
from models.AssetModel import AssetModel
from models import AssetTypeEnum
from models import DataChunkSchema

import aiofiles
import logging

logger = logging.getLogger('uvicorn.error')

data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1_data", "data"]
)

@data_router.post("/upload/{project_id}")
async def upload_data(request: Request, project_id: str, file: UploadFile,
                app_settings: Settings = Depends(get_settings)):

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)

    data_controller = DataController()
    is_valid, result_signal = data_controller.validate_uploaded_file(file)

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content = {"signal": result_signal}
        )

    file_path, file_id = data_controller.generate_unique_filepath(
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
            content = {"signal": ResponseSignal.FILE_UPLOAD_FAILED.value}
        )

    # store the assets in the database
    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)
    asset_record = Asset(
        asset_project_id= project.id,
        asset_type=AssetTypeEnum.FILE.value,
        asset_name=file_id, 
        asset_size=os.path.getsize(file_path),
        asset_config={} 
    )
    asset_record = await asset_model.create_asset(asset=asset_record)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseSignal.FILE_UPLOAD_SUCESS.value,
            "file_id": str(asset_record.id), 
        },
    )
    
@data_router.post("/process/{project_id}")
async def process_endpoint(request: Request, project_id: str, process_request: ProcessRequest):
    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size

    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    chunk_model = await ChunkModel.create_instance(db_client=request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)
    
    process_controller = ProcessController(project_id=project_id)
    
    asset_model = await AssetModel.create_instance(db_client=request.app.db_client)

    # Logic to identify which files to process
    project_file_ids = {}
    if process_request.file_id:
        asset_record = await asset_model.get_asset_record(
            asset_project_id=project.id,
            asset_name=process_request.file_id
        )
        if asset_record:
            project_file_ids = {asset_record.id: asset_record.asset_name}
        else:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"signal": ResponseSignal.FILE_NOT_FOUND_IN_PROJECT.value}
    )
    else:
        # Retrieve all assets for the project if no specific file_id is provided
        
        project_assets = await asset_model.get_all_project_assets(
            project_id=project.id,
            asset_type=AssetTypeEnum.FILE.value
        )
        # Using dot notation thanks to Pydantic mapping
        project_file_ids = {asset.id: asset.asset_name for asset in project_assets}
    
    if not project_file_ids:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.NO_FILES_IN_PROJECT.value}
        )
    
    # Optional reset of previous chunks
    if process_request.do_reset == 1:
        await chunk_model.delete_chunks_by_project_id(project_id=project.id)

    total_inserted = 0
    # Process each file identified in the list
    # Iterate over file IDs and names
    for asset_id, file_name in project_file_ids.items():
        file_content = process_controller.get_file_content(file_id=file_name)

        if not file_content:
            logger.warning(f"Skipping file {file_name}: unable to load content.")
            continue

        file_chunks = process_controller.process_file_content(
            file_content=file_content,
            file_id=file_name,
            chunk_size=chunk_size,
            chunk_overlap=overlap_size
        )

        if file_chunks:
            file_chunks_records = [
                DataChunkSchema(
                    chunk_text=chunk.page_content,
                    chunk_metadata=chunk.metadata,
                    chunk_order=i + 1,
                    chunk_project_id=project.id,
                    chunk_asset_id=asset_id
                )
                for i, chunk in enumerate(file_chunks)
            ]
            # Bulk insert chunks into MongoDB
            total_inserted += await chunk_model.insert_many_chunks(chunks=file_chunks_records)

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseSignal.PROCESSING_SUCESS.value,
            "number_of_chunks": total_inserted,
            "processed_files": len(project_file_ids)
        },
    )