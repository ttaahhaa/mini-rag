from fastapi import FastAPI, APIRouter, Depends, UploadFile, status, Request
from fastapi.responses import JSONResponse
import os
from helpers.config import get_settings, Settings
from controllers import DataController, ProjectController, ProcessController
from .schemas import ProcessRequest

from models import DataChunkSchema
from models import ProjectModel, ChunkModel
from models import ResponseSignal

import aiofiles
import logging

logger = logging.getLogger('uvicorn.error')
data_router = APIRouter(
    prefix="/api/v1/data",
    tags=["api_v1_data", "data"]
)

@data_router.post("/upload/{project_id}")
async def upload_data(requset: Request, project_id: str, file: UploadFile,
                app_settings: Settings = Depends(get_settings)):
    """
    Handle file upload for a given project, create the project if it does not exist,
    and persist the uploaded file to the project directory.

    The endpoint:
    - Ensures a `Project` document exists for the provided `project_id`, creating one if needed.
    - Validates the uploaded file using `DataController.validate_uploaded_file`.
    - Generates a unique file path and ID under the project directory.
    - Streams the file to disk in chunks of size `FILE_DEFAULT_CHUNK_SIZE` to avoid loading it fully in memory.
    - Returns:
        * 200 with a success signal and `file_id` when the file is stored successfully.
        * 400 with an appropriate failure signal if validation fails or an I/O error occurs.

    Args:
        request: The incoming HTTP request, used to access the shared database client.
        project_id: Logical identifier of the project to which the file will belong.
        file: The uploaded file payload (multipart/form-data).
        app_settings: Application settings dependency providing configuration such as chunk size.

    Returns:
        JSONResponse: A JSON response containing a status `signal` and, on success, the generated `file_id`.
    """
    project_model = ProjectModel(db_client=requset.app.db_client)
    print(f"Project Model Initialized: {project_model}")

    project = await project_model.get_project_or_create_one(project_id=project_id)
    print(f"Project Retrieved or Created: {project}")

    data_controller = DataController()
    is_valid, result_signal = data_controller.validate_uploaded_file(file)
    print(f"File Validation Result - is_valid: {is_valid}, signal: {result_signal}")

    if not is_valid:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
                content = {
                "signal": result_signal
            }
        )

    project_dir_path = ProjectController().get_project_path(project_id=project_id)
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
                content = {
                "signal": ResponseSignal.FILE_UPLOAD_FAILED.value
            }
        )
 

    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseSignal.FILE_UPLOAD_SUCESS.value,
            "file_id": file_id,
        },
    )
    
@data_router.post("/process/{project_id}")
async def process_endpoint(request: Request,project_id: str, process_request: ProcessRequest):
    file_id = process_request.file_id
    chunk_size = process_request.chunk_size
    overlap_size = process_request.overlap_size

    project_model = ProjectModel(db_client=request.app.db_client)
    chunk_model = ChunkModel(db_client=request.app.db_client)

    project = await project_model.get_project_or_create_one(project_id=project_id)
    process_controller = ProcessController(project_id=project_id)
    
    file_content = process_controller.get_file_content(file_id=file_id)

    if file_content is None or len(file_content) == 0 or all(len(rec.page_content.strip()) == 0 for rec in file_content):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.NO_TEXT_IN_FILE.value}
        )

    file_chunks = process_controller.process_file_content(file_content=file_content,
                                                            file_id=file_id,
                                                            chunk_size=chunk_size,
                                                            chunk_overlap=overlap_size)

    if file_chunks is None or len(file_chunks) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
                content = {
                "signal": ResponseSignal.PROCESSING_FAILED.value
            }
        )

    file_chunks_records = [
         DataChunkSchema(
            chunk_text=chunk.page_content,
            chunk_metadata=chunk.metadata,
            chunk_order=i + 1,
            chunk_project_id=project.id)
         for i, chunk in enumerate(file_chunks)
    ]

    if process_request.do_reset ==1:
        deleted_chunks = await chunk_model.delete_chunks_by_project_id(project_id=project.id)
        print("++++++++++++++++++++++++++++")
        print(f"Deleted {deleted_chunks} existing chunks for project_id: {project_id}")

    numbers_of_inserted_chunks = await chunk_model.insert_many_chunks(chunks=file_chunks_records)

    if numbers_of_inserted_chunks == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
                content = {
                "signal": ResponseSignal.PROCESSING_FAILED.value
            }
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "signal": ResponseSignal.PROCESSING_SUCESS.value,
            "number_of_chunks": numbers_of_inserted_chunks,
        },
    )
