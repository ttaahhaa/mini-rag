import os
import asyncio
from .BaseAsyncController import BaseAsyncController
from .ProjectAsyncController import ProjectAsyncController
from models import ProcessingEnum
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class ProcessAsyncController(BaseAsyncController):
    def __init__(self, project_id: int):
        super().__init__()
        self.project_id = project_id
        # We store the task to get the path, but we'll await it inside methods
        self.project_controller = ProjectAsyncController()

    async def get_file_loader(self, file_id: str):
        """Async: Checks path and returns the correct LangChain loader."""
        project_path = await self.project_controller.get_project_path(self.project_id)
        file_path = os.path.join(project_path, file_id)
        
        if not await asyncio.to_thread(os.path.exists, file_path):
            return None
        
        file_ext = os.path.splitext(file_id)[-1].lower()
        if file_ext == ProcessingEnum.TXT.value:
            return TextLoader(file_path=file_path, encoding='utf-8')
        if file_ext == ProcessingEnum.PDF.value:
            return PyMuPDFLoader(file_path=file_path)
        
        return None
    
    async def get_file_content(self, file_id: str):
        """Async: Offloads blocking document loading to a worker thread."""
        loader = await self.get_file_loader(file_id=file_id)
        if loader:
            # LangChain's .load() is blocking; run in thread to keep Event Loop free
            return await asyncio.to_thread(loader.load)
        return None

    async def process_file_content(self, file_content: list, **kwargs):
        """Async: Offloads CPU-intensive text splitting to a worker thread."""
        # CPU-heavy operations like string splitting can block the entire API
        return await asyncio.to_thread(self._sync_split, file_content, **kwargs)

    def _sync_split(self, file_content, chunk_size=100, chunk_overlap=20):
        """Internal synchronous helper for the thread worker."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        texts = [rec.page_content for rec in file_content]
        metadatas = [rec.metadata for rec in file_content]

        return text_splitter.create_documents(texts, metadatas=metadatas)