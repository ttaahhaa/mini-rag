from enum import Enum

class LLMEnums(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    QDRANT = "QDRANT"
    AsyncQDRANT = "AsyncQDRANT"
    MILVUS = "MILVUS"
    AsyncMILVUS = "AsyncMILVUS"

class OpenAIEnums(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class CohereEnums(Enum):
    SYSTEM = "SYSTEM"
    USER = "USER"
    ASSISTANT = "CHATBOT"
    
    DOCUMENT = "search_document" 
    QUERY = "search_query"

class DocumentTypeEnum(Enum):
    DOCUMENT = "document"
    QUERY = "query"