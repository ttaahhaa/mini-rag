import os
import asyncio
from .BaseAsyncController import BaseAsyncController

class ProjectAsyncController(BaseAsyncController):
    def __init__(self):
        super().__init__()
    
    async def get_project_path(self, project_id: str):
        """Async: OS directory creation is a blocking I/O operation."""
        project_dir = os.path.join(self.file_dir, str(project_id))
        if not await asyncio.to_thread(os.path.exists, project_dir):
            await asyncio.to_thread(os.makedirs, project_dir, exist_ok=True)
        return project_dir