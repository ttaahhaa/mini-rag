from .BaseController import BaseController
from models import ResponseSignal
from models.db_schemas import ProjectSchema
class NLPController(BaseController):
    def __init__(self, generation_client, embedding_client, vectordb_client):
        super().__init__()

        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.vectordb_client = vectordb_client


    def create_collection_name(self, project_id: str):
        return f"Collection_{project_id}".strip()
    
    def reset_vector_db_collection(self, project: ProjectSchema):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)
    
    def get_vector_collection_info(self, collection_name, project: ProjectSchema):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)
        return collection_info
    
    