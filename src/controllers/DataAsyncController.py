import re
import os
import asyncio
from fastapi import UploadFile
from .BaseAsyncController import BaseAsyncController
from .ProjectAsyncController import ProjectAsyncController
from models import ResponseSignal

class DataAsyncController(BaseAsyncController):
    def __init__(self):
        super().__init__()
        self.file_scale = 1048576 # 1MB
        
    def validate_uploaded_file(self, file: UploadFile):
        """Sync: Purely checking metadata already in memory."""
        if file.content_type not in self.get_settings.FILE_ALLOWED_TYPES:
            return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value
        
        if file.size > self.get_settings.FILE_MAX_SIZE * self.file_scale:
            return False, ResponseSignal.FILE_SIZE_EXCEEDED.value
        
        return True, ResponseSignal.FILE_VALIDATED_SUCESS.value
        
    async def generate_unique_filepath(self, org_file_name: str, project_id: int):
        """Async: Checks file existence on disk."""
        project_path = await ProjectAsyncController().get_project_path(project_id=project_id)
        cleaned_filename = self.get_clean_file_name(org_file_name)

        while True:
            random_key = self.generate_random_string()
            filename = f"{random_key}_{cleaned_filename}"
            new_file_path = os.path.join(project_path, filename)

            # Non-blocking check for file existence
            if not await asyncio.to_thread(os.path.exists, new_file_path):
                return new_file_path, filename
    
    def get_clean_file_name(self, org_file_name: str):
        """Sync: Fast regex/string manipulation."""
        cleaned = re.sub(r'[^\w.]', '', org_file_name.strip())
        return cleaned.replace(" ", "_")