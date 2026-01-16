from enum import Enum 

class LLMEnums(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    CUSTOM = "custom"

class OpenAIEnums(Enum):
    # Models
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4_TURBO   = "gpt-4-turbo"   # or current recommended GPT‑4.x id
    GPT_4O_MINI   = "gpt-4o-mini"

    # Message Roles
    SYSTEM = "system"
    USER   = "user"
    ASSISTANT = "assistant"

class CohereEnums(Enum):
    # Message Roles
    SYSTEM = "system"
    USER   = "user"
    ASSISTANT = "chatbot"
    
    DOCUMENT="search_document"
    QUERY="search_query"
    
