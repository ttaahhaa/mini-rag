from ..LLMInterface import LLMInterface
from openai import OpenAI
from helpers.config import get_settings
import logging
from LLMEnums import OpenAIEnums

class OpenAIProvidor(LLMInterface):
    def __init__(self, api_key: str = None, api_url: str = None,
                  default_input_max_tokens: int = 1000, defualt_output_max_tokens: int = 1000,
                  temperature: float = 0.1):
        
        self.api_key = api_key or get_settings().OPENAI_API_KEY
        self.api_url = api_url or get_settings().OPENAI_API_URL

        self.default_input_max_tokens = default_input_max_tokens
        self.defualt_output_max_tokens = defualt_output_max_tokens
        self.default_generation_temperature = temperature
        
        self.generation_model_id = None 
        self.embedding_model_id = None
        self.embeding_size = None

        self.client = OpenAI(api_key=self.api_key, api_base=self.api_url)
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id
        self.logger.info(f"Set generation model to {model_id}")
    
    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embeding_size = embedding_size
        self.logger.info(f"Set embedding model to {model_id} with size {embedding_size}")

    def process_text(self, text: str) -> str:
        return text[:self.default_input_max_tokens].strip()
    
    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int = None, temperature: float = None) -> str:
        if not self.client:
            self.logger.info(f"OpenAI Client was not set")
            return None
        
        if not self.generation_model_id:
            self.logger.info(f"Generation model for OPENAI was not set")
            return None
        
        max_output_tokens = max_output_tokens if max_output_tokens else self.defualt_output_max_tokens
        temperature = temperature if temperature else self.default_generation_temperature

        chat_history.append(self.construct_prompt(prompt, role=OpenAIEnums.USER.value))
        response = self.client.chat.completions.create(
            model=self.generation_model_id,
            messages=chat_history,
            max_tokens=max_output_tokens,
            temperature=temperature
        )
        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            self.logger.error(f"Failed to get response from OpenAI for prompt")
            return None
        return response.choices[0].message.content

    def embed_text(self, text, document_type = None) -> list[float]:
        if not self.client:
            self.logger.info(f"OpenAI Client was not set")
            return None
        
        if not self.embedding_model_id:
            self.logger.info(f"Embedding model for OPENAI was not set")
            return None
        
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model_id
        )

        if not response or not response.data or len(response.data) == 0 or not response.data[0].embedding:
            self.logger.error(f"Failed to get embedding from OpenAI for text: {text}")
            return None 
        
        return response.data[0].embedding
    def construct_prompt(self, prompt: str, role: str = "user") -> dict:
        return {
            "role": role,
            "content": self.process_text(prompt)
        }
  
        
    