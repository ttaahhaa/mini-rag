from ..VectorDBInterface import VectorDBInterface
from qdrant_client import QdrantClient

class QDrantProvider(VectorDBInterface):
    def __init__(self, ):
        self.client = QdrantClient(host=host, port=port)

    def connect(self):
        pass

    def disconnect(self):
        pass

    def is_collection_exists(self, collection_name):
        pass

    def create_collection(self, collection_name: str, embedding_size: int,
                           do_reset: bool = False):
        pass

    def insert_one(self, collection_name: str, text: str,
                   vector: list, metadata: dict,
                   record_id: str = None):
        pass

    def insert_many(self, collection_name: str, texts: list,
                    vectors: list, metadatas: list,
                    record_ids: list = None, batch_size: int = 5):
        pass

    def get_collection_info(self, collection_name):
        pass

    def delete_collection(self, collection_name):
        pass