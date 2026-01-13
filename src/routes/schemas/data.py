from pydantic import BaseModel
from typing import Optional

class ProcessRequest(BaseModel):
    file_id: str | None = None # None means it is optional
    chunk_size: int = 100       # Optional in JSON, but MUST be an int if sent
    overlap_size: int = 20     # Optional in JSON, but MUST be an int if sent
    do_reset: int = 0  # Optional in JSON, but MUST be an int if sent