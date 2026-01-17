from .BaseController import BaseController
from models import ResponseSignal

class NLPController(BaseController):
    def __init__(self, generation_client, embedding_client, vectordb_client):
        super().__init__()

        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.vectordb_client = vectordb_client

        
