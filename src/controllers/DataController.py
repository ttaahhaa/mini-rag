from fastapi import UploadFile
from .BaseController import BaseController
from .ProjectController import ProjectController
from models import ResponseSignal
import re
import os 

class DataController(BaseController):
    def __init__(self):
        super().__init__()
        self.file_scale = 1048576
        
    def validate_uploaded_file(self, file: UploadFile):
        # types
        if file.content_type not in self.get_settings.FILE_ALLOWED_TYPES:
            return False, ResponseSignal.FILE_TYPE_NOT_SUPPORTED.value
        # size
        if file.size > self.get_settings.FILE_MAX_SIZE * self.file_scale:
            return False, ResponseSignal.FILE_SIZE_EXCEEDED.value
        
        return True, ResponseSignal.FILE_VALIDATED_SUCESS.value
        
    def generate_unique_filepath(self, org_file_name: str, project_id: str):
        project_path = ProjectController().get_project_path(project_id=project_id)
        cleaned_filename = self.get_clean_file_name(
            org_file_name=org_file_name
        )

        while True:
            random_key = self.generate_random_string()
            filename = f"{random_key}_{cleaned_filename}"
            new_file_path = os.path.join(project_path, filename)

            if not os.path.exists(new_file_path):
                return new_file_path, filename
    
    def get_clean_file_name(self, org_file_name: str):
        # remove any special characters, except underscore and .
        cleaned_file_name = re.sub(r'[^\w.]', '', org_file_name.strip())

        # replace spaces with inderscores
        cleaned_file_name = cleaned_file_name.replace(" ", "_")

        return cleaned_file_name