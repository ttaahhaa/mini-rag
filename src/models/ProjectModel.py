from .BaseDataModel import BaseDataModel
from models.db_schemas import Project  # Ensure this is now a SQLAlchemy ORM model
from .enums.DataBaseEnums import DatabaseEnum
from sqlalchemy.future import select
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

class ProjectModel(BaseDataModel):
    def __init__(self, db_session: AsyncSession):
        """
        In SQLAlchemy, db_client is usually an async_sessionmaker.
        """
        super().__init__(db_session=db_session)

    @classmethod
    async def create_instance(cls, db_session: AsyncSession):
        """
        Factory method to create an instance using the injected session.
        """
        return cls(db_session)

    async def create_project(self, project: Project):
            """
            Minimal: Dependency handles the begin/commit.
            """
            self.db_session.add(project)
            await self.db_session.flush() 
            return project

    async def get_project_or_create_one(self, project_id: int) -> Project:
            """
            Uses the shared session. No more 'async with session.begin()'.
            """
            # 1. Try to find the project
            query = select(Project).where(Project.project_id == project_id)
            result = await self.db_session.execute(query)
            project = result.scalar_one_or_none()

            # 2. If not found, create a new record
            if project is None:
                new_project = Project(project_id=project_id)
                self.db_session.add(new_project)
                # Flush ensures the project is sent to the DB (assigning IDs, etc.)
                # but the COMMIT is still handled by the get_db helper.
                await self.db_session.flush() 
                await self.db_session.refresh(new_project)
                return new_project
            
            return project

    async def get_all_projects(self, page: int = 1, page_size: int = 10):
        """
        Paginated retrieval using the active session.
        """
        # 1. Count total documents
        count_query = select(func.count(Project.project_id))
        count_result = await self.db_session.execute(count_query)
        total_documents = count_result.scalar_one()

        # 2. Calculate total pages
        total_pages = (total_documents + page_size - 1) // page_size

        # 3. Fetch paginated data
        query = (
            select(Project)
            .offset((page - 1) * page_size)
            .limit(page_size)
        )
        result = await self.db_session.execute(query)
        projects = result.scalars().all()

        return projects, total_pages