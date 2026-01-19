from enum import Enum

class TemplateLanguagesEnums(Enum):
    En= "en"
    Ar= "ar"
    

class TemplateDirectoriesAndFilesEnums(Enum):
    LOCALES = "locales"
    STORES = "stores"
    TEMPLATES = "templates"
    RAG = "rag"

class PromptsVariables(Enum):
    SYSTEM_PROMPT = "system_prompt"
    DOCUMENT_PROMPT = "document_prompt"
    FOOTER_PROMPT = "footer_prompt"