

class BaseDataModel:
    def __init__(self, db_client):
        self.db_client = db_client
        from helpers.config import get_settings
        self.app_settings = get_settings()
        