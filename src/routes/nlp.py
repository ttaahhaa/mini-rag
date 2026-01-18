from fastapi import FastAPI, APIRouter, Depends, status, Request
from fastapi.responses import JSONResponse
import os
from controllers import NLPController
from helpers.config import get_settings, Settings
from routes.schemas.nlp import PushRequest, SearchRequest
from models.ProjectModel import ProjectModel
from models import ChunkModel, ResponseSignal

import logging

# Initialize logger to capture errors/info in the Uvicorn console
logger = logging.getLogger('uvicorn.error')

# Define the router with a standard prefix and tags for Swagger documentation
nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1_nlp", "nlp"]
)

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: str,
                        push_request: PushRequest):
    """
    Endpoint: Processes document chunks from MongoDB and pushes them into the Vector Database.
    Includes pagination logic to handle large projects without memory overflow.
    """
    
    # Initialize database models using the app's global database client
    project_model = await ProjectModel.create_instance(db_client=request.app.db_client)
    chunk_model = ChunkModel(db_client=request.app.db_client)

    # Fetch existing project metadata or create a new entry if it doesn't exist
    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal": ResponseSignal.PROJECT_NOT_FOUND_ERROR.value}
            )
    
    # Initialize the NLP Controller with necessary AI and Database clients
    # These clients (generation, embedding, vectordb) are pre-configured in main.py
    nlp_controller = NLPController(
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
        vectordb_client=request.app.vectordb_client
    )

    # Fetch initial chunks to verify project content (Optional check)
    chunks = await chunk_model.get_all_chunks_in_a_project(
        project_id=project.id
    )
    
    has_records = True
    page_no = 1
    inserted_items_count = 0
    idx = 0  # Global index used to generate unique point IDs for the vector database

    # START PAGINATION LOOP: Processes 50 chunks at a time (default page size)
    while has_records:
        page_chunks = await chunk_model.get_all_chunks_in_a_project(
            project_id=project.id, 
            page_no=page_no
        )
        
        # If chunks are found on the current page, move to the next page for the next iteration
        if len(page_chunks):
            page_no += 1
        
        # Exit loop if no more chunks are found in MongoDB
        if not page_chunks or len(page_chunks) == 0:
            has_records = False
            break
            
        # Generate a list of IDs (e.g., [0, 1, 2...]) to uniquely identify each vector in the collection
        chunks_ids = list(range(idx, idx + len(page_chunks)))
        idx += len(page_chunks)    

        # Pass the batch of chunks to the controller for embedding and vector database insertion
        is_inserted = nlp_controller.index_into_vector_db(
            project=project,
            chunks=page_chunks,
            do_reset=push_request.do_reset,
            chunks_ids=chunks_ids
        )
        
        # Handle failures during vector DB insertion
        if not is_inserted:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseSignal.INSERT_INTO_VECTORDB_ERROR.value
                }
            )
        inserted_items_count += len(page_chunks)

    return JSONResponse(
        content={
            "signal": ResponseSignal.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": inserted_items_count
        }
    )

@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: str):
    """
    Endpoint: Retrieves metadata about a specific project's vector collection.
    Useful for checking the number of indexed points or collection status.
    """

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
    )

    # Get stats from the vector database (e.g., Qdrant collection info)
    collection_info = nlp_controller.get_vector_collection_info(project=project,)

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info
        }
    )

@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: str, search_request: SearchRequest):
    """
    Endpoint: Performs Semantic Search. 
    Converts user input text into a vector and finds similar vectors in the project's collection.
    """

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController(
        vectordb_client=request.app.vectordb_client,
        generation_client=request.app.generation_client,
        embedding_client=request.app.embedding_client,
    )

    # Perform the actual semantic search
    results = nlp_controller.search_vector_db_collection(
        project=project, 
        text=search_request.text, 
        limit=search_request.limit
    )

    if not results:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseSignal.VECTORDB_SEARCH_ERROR.value,
            }
        )

    return JSONResponse(
        content={
            "signal": ResponseSignal.VECTORDB_SEARCH_SUCCESS.value,
            "results": results
        }
    )