# helpers/db.py
from fastapi import Request

async def get_db(request: Request):
    # Create a session from the factory stored in app
    async with request.app.db_client() as session:
        # Start the transaction once here
        async with session.begin():
            yield session
        # Commit happens here automatically if no exception is raised