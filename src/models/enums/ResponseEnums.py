from enum import Enum

class ResponseSignal(Enum):
    
    FILE_VALIDATED_SUCESS = "file_validate_sucessfully"
    FILE_TYPE_NOT_SUPPORTED = "file_type_is_not_supported"
    FILE_SIZE_EXCEEDED = "file_size_exceeded"
    FILE_UPLOAD_SUCESS = "file_upload_success"
    FILE_UPLOAD_FAILED = "file_upload_failed"

    PROCESSING_SUCESS = "processing_success"
    PROCESSING_FAILED = "processing_failed"
    NO_TEXT_IN_FILE = "no_text_in_file"

    NO_FILES_IN_PROJECT = "no_files_in_project"
    FILE_NOT_FOUND_IN_PROJECT = "file_not_found_in_project"