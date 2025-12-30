from enum import Enum

class ResponseSignal(Enum):
    
    FILE_VALIDATED_SUCESS = "file_validate_sucessfully"
    FLE_TYPE_NOT_SUPPORTED = "file_type_is_not_supported"
    FILE_SIZE_EXCEEDED = "file_size_exceeded"
    FILE_UPLOAD_SUCESS = "file_upload_success"
    FILE_UPLOAD_FAILED = "file_upload_failed"