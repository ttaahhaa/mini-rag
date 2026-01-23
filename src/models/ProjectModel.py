from .BaseDataModel import BaseDataModel
from models.db_schemas import Project  # Ensure this is now a SQLAlchemy ORM model
from .enums.DataBaseEnums import DatabaseEnum
from sqlalchemy.future import select
from sqlalchemy import func

class ProjectModel(BaseDataModel):
    def __init__(self, db_client: object):
        """
        In SQLAlchemy, db_client is usually an async_sessionmaker.
        """
        super().__init__(db_client=db_client)
        self.db_client = db_client

    @classmethod
    async def create_instance(cls, db_client: object):
        """
        Factory method to create an instance of ProjectModel.
        SQLAlchemy handles table creation elsewhere, so we remove init_collection.
        """
        instance = cls(db_client)
        return instance

    async def create_project(self, project: Project):
        """
        Inserts a new project into the SQL database.
        """
        async with self.db_client() as session:
            async with session.begin():
                session.add(project)
            # Commit is handled by the 'begin' context or explicitly
            await session.commit()
            await session.refresh(project)
        
        return project

    async def get_project_or_create_one(self, project_id: int) -> Project:
        """
        Queries for a project by ID. If it doesn't exist, creates it.
        """
        async with self.db_client() as session:
            # 1. Try to find the project
            query = select(Project).where(Project.project_id == project_id)
            result = await session.execute(query)
            project = result.scalar_one_or_none()

            # 2. If not found, create a new record
            if project is None:
                # We open a transaction block only for the creation part
                async with session.begin():
                    new_project = Project(project_id=project_id)
                    session.add(new_project)
                    # Flush sends the data to the DB so refresh can work, 
                    # but doesn't close the transaction yet.
                    await session.flush() 
                
                # Now that the transaction is committed by exiting the 'begin' block, 
                # we can refresh the object.
                await session.refresh(new_project)
                return new_project
            
            return project

    async def get_all_projects(self, page: int = 1, page_size: int = 10):
        """
        Retrieves a paginated list of projects and the total page count.
        """
        async with self.db_client() as session:
            async with session.begin():
                # 1. Count total documents for pagination
                count_query = select(func.count(Project.project_id))
                count_result = await session.execute(count_query)
                total_documents = count_result.scalar_one()

                # 2. Calculate total pages
                total_pages = total_documents // page_size
                if total_documents % page_size > 0:
                    total_pages += 1

                # 3. Fetch paginated data using offset and limit
                query = (
                    select(Project)
                    .offset((page - 1) * page_size)
                    .limit(page_size)
                )
                result = await session.execute(query)
                projects = result.scalars().all()

                return projects, total_pages