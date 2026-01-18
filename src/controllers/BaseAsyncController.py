import os
import random
import string
import asyncio
from helpers.config import get_settings

class BaseAsyncController:
    def __init__(self):
        self.get_settings = get_settings()
        self.base_dir = os.path.dirname(os.path.dirname(__file__))
        self.file_dir = os.path.join(self.base_dir, 'assets', 'files')
        self.database_dir = os.path.join(self.base_dir, "assets", "database")

    def generate_random_string(self, length: int = 12):
        """Sync is fine: purely CPU-bound and ultra-fast."""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    
    async def get_database_path(self, db_name: str):
        """Async: Creating directories involves OS I/O."""
        database_path = os.path.join(self.database_dir, db_name)
        if not await asyncio.to_thread(os.path.exists, database_path):
            await asyncio.to_thread(os.makedirs, database_path, exist_ok=True)
        return database_path