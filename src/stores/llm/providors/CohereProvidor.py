from ..LLMInterface import LLMInterface
from ..LLMEnums import CohereEnums, LLMEnums
from helpers.config import get_settings
import cohere
import logging

class CohereProvidor(LLMInterface):
    def __init__(self, api_key: str = None, default_input_max_tokens: int = 1000,
                   defualt_output_max_tokens: int = 1000,
                  temperature: float = 0.1):
            self.api_key = api_key or get_settings().OPENAI_API_KEY

            self.default_input_max_tokens = default_input_max_tokens
            self.defualt_output_max_tokens = defualt_output_max_tokens
            self.default_generation_temperature = temperature

            self.generation_model_id = None 
            self.embedding_model_id = None
            self.embeding_size = None

            self.client = cohere.Client(self.api_key)
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

    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int = None,
                       temperature: float = None) -> str:
        if not self.client:
            self.logger.info(f"OpenAI Client was not set")
            return None
        
        if not self.generation_model_id:
            self.logger.info(f"Generation model for OPENAI was not set")
            return None
        
        response = self.client.chat(
                model=self.generation_model_id,
                chat_history=chat_history,
                message=self.construct_prompt(prompt),
                max_tokens=max_output_tokens if max_output_tokens else self.defualt_output_max_tokens,
                temperature=temperature if temperature else self.default_generation_temperature
        )

        if not response or not response.text:
            self.logger.error(f"Failed to get response from Cohere for prompt")
            return None
        return response.text

    def embed_text(self, text, document_type = None) -> list[float]:
        if not self.client:
            self.logger.error(f"OpenAI Client was not set")
            return None
        
        if not self.embedding_model_id:
            self.logger.error(f"Embedding model for OPENAI was not set")
            return None
        
        input_type = CohereEnums.DOCUMNET
        if document_type == LLMEnums.QUERY:
            input_type = CohereEnums.QUERY
        
        response = self.client.embed(
            model=self.embedding_model_id,
            texts=[self.process_text(text)],
            input_type=input_type,
            embedding_types=["float"]
        )

        if not response or not response.embeddings or response.embeddings.float:
            self.logger.error(f"Failed to get embeddings from Cohere for text")
            return None
        return response.embeddings.float[0]

    def construct_prompt(self, prompt: str, role: str = "user") -> dict:
        return {
            "role": role,
            "content": self.process_text(prompt)
        }
