from .BaseAsyncController import BaseAsyncController
from models import ResponseSignal
from models.db_schemas import Project, DataChunk
from stores.llm.LLMEnums import DocumentTypeEnum
from typing import List
import json 
import asyncio
from stores.templates import TemplateParser
from models.enums.TemplatesEnum import TemplateDirectoriesAndFilesEnums, PromptsVariables

class NLPAsyncController(BaseAsyncController):
    def __init__(self, generation_client, embedding_client, vectordb_client, template_parser: TemplateParser):
        """
        Initializes the Async NLP Controller.
        """
        super().__init__()
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.vectordb_client = vectordb_client
        self.template_parser = template_parser

    def create_collection_name(self, project_id: int):
        return f"Collection_{project_id}".strip()
    
    async def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return await self.vectordb_client.delete_collection(collection_name=collection_name)
    
    async def get_vector_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = await self.vectordb_client.get_collection_info(collection_name=collection_name)
        
        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
    async def index_into_vector_db(self, project: Project, 
                                   chunks: List[DataChunk], 
                                   chunks_ids: List[int], 
                                   do_reset: bool = False):
        """
        Handles the full indexing pipeline asynchronously.
        """
        collection_name = self.create_collection_name(project_id=project.project_id)
        texts = [c.chunk_text for c in chunks]
        metadata = [i.chunk_metadata for i in chunks]

        # 1. Offload heavy embedding to a worker thread
        vectors = await asyncio.to_thread(
            self.embedding_client.embed_batch,
            texts=texts,
            document_type=DocumentTypeEnum.DOCUMENT.value
        )

        if not vectors:
            self.logger.error("Indexing failed: No vectors generated.")
            return False

        # 2. Create collection (Async database call)
        await self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset
        )

        # 3. High-throughput async insertion
        result = await self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            vectors=vectors,
            metadatas=metadata,
            record_ids=chunks_ids
        )

        return result

    async def search_vector_db_collection(self, project: Project, text: str, limit: int = 10):
        """
        Asynchronous semantic search.
        """
        
        collection_name = self.create_collection_name(project_id=project.project_id)

        # 1. Offload single-text embedding to a worker thread
        # This fixes the "TypeError: object list can't be used in 'await' expression"
        vector = await asyncio.to_thread(
            self.embedding_client.embed_text,
            text=text,
            document_type=DocumentTypeEnum.QUERY.value
        )

        if not vector or len(vector) == 0:
            return []

        # 2. Execute the async search against the vector store
        results = await self.vectordb_client.search_by_vector(
            collection_name=collection_name,
            vector=vector,
            limit=limit
        )

        return results if results else []
    
    async def asnwer_rag_question(self, project: Project, query: str, limit: int = 10):
        answer, full_prompt, chat_history = None, None, None
        
        # Step 1: Retrieve related documents (FIX: Added await)
        retrieved_documents = await self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit
        )
        
        if not retrieved_documents:
            return answer, full_prompt, chat_history

        # Step 2: Construct Prompt LLM (FIX: Added awaits for async template parser)
        system_prompt = await self.template_parser.get(
            TemplateDirectoriesAndFilesEnums.RAG.value,
            PromptsVariables.SYSTEM_PROMPT.value # Ensure this key exists in your enums
        )
        
        # Process document templates asynchronously for maximum speed
        doc_template_tasks = [
            self.template_parser.get(
                TemplateDirectoriesAndFilesEnums.RAG.value,
                PromptsVariables.DOCUMENT_PROMPT.value, 
                {"doc_num": idx + 1, "chunk_text": self.generation_client.process_text(doc.text)}
            )
            for idx, doc in enumerate(retrieved_documents)
        ]
        
        # Execute all string substitutions in parallel
        doc_prompts_list = await asyncio.gather(*doc_template_tasks)

        # Clean up the list to remove any None values returned by the parser
        filtered_prompts = [p for p in doc_prompts_list if p is not None]

        if not filtered_prompts:
            documents_prompts = "No relevant context found."
        else:
            documents_prompts = "\n".join(filtered_prompts)

        footer_prompt = await self.template_parser.get(
            TemplateDirectoriesAndFilesEnums.RAG.value,
            PromptsVariables.FOOTER_PROMPT.value,{"query": query}
        )
        
        # Construct chat history using provider-specific roles
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value
            )
        ]

        full_prompt = "\n\n".join([documents_prompts, footer_prompt])

        # Step 3: Generate Answer (Offload blocking sync LLM call to a thread)
        answer = await asyncio.to_thread(
            self.generation_client.generate_text,
            prompt=full_prompt,
            chat_history=chat_history
        )

        return answer, full_prompt, chat_history