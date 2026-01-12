from .BaseDataModel import BaseDataModel
from .db_schemas.project import ProjectSchema
from .enums.DataBaseEnums import DatabaseEnum

class ProjectModel(BaseDataModel):
    def __init__(self, db_client: object):
        super().__init__(db_client)
        self.collection = self.db_client[DatabaseEnum.COLLECTION_PROJECT_NAME.value]

    @classmethod
    async def create_instance(cls, db_client: object):
        """
        Factory method to create an instance of ProjectModel and initialize the collection.
        
        :param db_client: The database client to interact with the database.
        :return: An initialized instance of ProjectModel.
        """
        instance = cls(db_client)
        await instance.init_collection()
        return instance

    async def init_collection(self):
        """
        Initialize the project collection in the database.
        This method checks if the collection exists, and if not, creates it
        along with the necessary indexes defined in the ProjectSchema.
        
        :param self: The instance of the ProjectModel class.
        :return: None
        """
        all_collections = await self.db_client.list_collection_names()
        if DatabaseEnum.COLLECTION_PROJECT_NAME.value not in all_collections:
            self.collection = self.db_client[DatabaseEnum.COLLECTION_PROJECT_NAME.value]
            indexes = ProjectSchema.get_indexes()
            for index in indexes:
                await self.collection.create_index(
                    index["key"],
                    name=index["name"],
                    unique=index["unique"],
                )

    
    async def create_project(self, project: ProjectSchema):
        # 1. Convert Pydantic model to dict
        # dump using aliases so Mongo sees _id if present
        project_dict = project.model_dump(by_alias=True, exclude_unset=True)

        # ensure we don’t send an _id if it's None
        project_dict.pop("_id", None)

        # 2. Insert into MongoDB
        result = await self.collection.insert_one(project_dict)

        # 3. Attach MongoDB _id to the model
        project.id = result.inserted_id  # <- use _id, not id

        # 4. Return updated project
        return project


    async def get_project_or_create_one(self, project_id: str) -> ProjectSchema:
        doc = await self.collection.find_one({"project_id": project_id})
        if doc:
            return ProjectSchema(**doc)  # doc has _id -> mapped to id via alias
        project = ProjectSchema(project_id=project_id)
        return await self.create_project(project)
    
    
    async def get_all_projects(self, page: int = 1, page_size: int = 10):
        # 1. Count total projects in the collection to calculate how many pages exist
        total_documents = await self.collection.count_documents({})
        
        # 2. Use integer division to find base number of pages (e.g., 25 // 10 = 2)
        total_pages = total_documents // page_size

        # 3. If there's a remainder, add one more page (e.g., 25 % 10 = 5, so we need 3 pages)
        if total_documents % page_size != 0:
            total_pages += 1
        
        # 4. skip(): Jumps over items from previous pages 
        #    limit(): Restricts the result to only the current page's count
        cursor = self.collection.find().skip((page - 1) * page_size).limit(page_size)
        
        projects = []
        # 5. Iterate through the database cursor asynchronously
        async for document in cursor:
            projects.append(
                # 6. Convert raw MongoDB dictionary into a validated Pydantic object
                ProjectSchema(**document)
            )
        
        # 7. Return both the list of projects and the total page count for the UI/Frontend
        return projects, total_pages