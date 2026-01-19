from ..LLMInterface import LLMInterface
from openai import OpenAI
from helpers.config import get_settings
import logging
from ..LLMEnums import OpenAIEnums
import time
import random

class OpenAIProvider(LLMInterface):
    """
    Concrete implementation of LLMInterface for OpenAI API integration.
    Supports both OpenAI's official API and OpenAI-compatible endpoints (via api_url).
    Provides text generation using chat completions and embedding capabilities.
    """
    
    def __init__(self, api_key: str = None, api_url: str = None,
                 default_input_max_tokens: int = 1000, default_output_max_tokens: int = 1000,
                 temperature: float = 0.1):
        # Load settings from environment configuration
        settings = get_settings()
        
        # API key from parameter or fallback to settings
        self.api_key = api_key or settings.OPENAI_API_KEY
        
        # API base URL - allows using OpenAI-compatible endpoints (e.g., Azure OpenAI, local models)
        self.api_url = api_url or settings.OPENAI_API_URL
        
        # Maximum tokens to process from input text (truncation limit)
        self.default_input_max_tokens = default_input_max_tokens
        
        # Maximum tokens the model can generate in response
        self.default_output_max_tokens = default_output_max_tokens
        
        # Controls randomness in generation (0.0 = deterministic, higher = more creative)
        self.default_generation_temperature = temperature
        
        # Model identifiers - set later via setter methods
        self.generation_model_id = None  # e.g., "gpt-4", "gpt-3.5-turbo"
        self.embedding_model_id = None   # e.g., "text-embedding-3-small"
        self.embedding_size = None
        
        # Initialize OpenAI client with API key and optional custom base URL
        self.client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        self.enums = OpenAIEnums
        # Logger for tracking operations and debugging
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        """
        Configure which OpenAI model to use for text generation.
        Called during app startup via LLMProviderFactory.
        
        Args:
            model_id: OpenAI model identifier (e.g., "gpt-4", "gpt-3.5-turbo")
        """
        self.generation_model_id = model_id
        self.logger.info(f"Set generation model to {model_id}")
    
    def set_embedding_model(self, model_id: str, embedding_size: int):
        """
        Configure which OpenAI model to use for embeddings.
        
        Args:
            model_id: OpenAI embedding model (e.g., "text-embedding-3-small", "text-embedding-ada-002")
            embedding_size: Vector dimensions (e.g., 1536 for ada-002, 1536 for text-embedding-3-small)
        """
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.logger.info(f"Set embedding model to {model_id} with size {embedding_size}")

    def process_text(self, text: str) -> str:
        """
        Preprocess text before sending to OpenAI API.
        Truncates to max input tokens and removes extra whitespace.
        
        Args:
            text: Raw input text
            
        Returns:
            Processed text within token limits
        """
        return text[:self.default_input_max_tokens].strip()
    
    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int = None, 
                     temperature: float = None) -> str:
        """
        Generate text using OpenAI's chat completions endpoint.
        
        Args:
            prompt: User message/question to generate response for
            chat_history: List of previous messages in OpenAI format [{"role": "user", "content": "..."}]
            max_output_tokens: Override default max output tokens
            temperature: Override default temperature for this generation
            
        Returns:
            Generated text response or None if error occurs
        """
        # Validate client is initialized
        if not self.client:
            self.logger.error(f"OpenAI Client was not set")  # FIXED: Changed from info to error
            return None
        
        # Validate generation model is configured
        if not self.generation_model_id:
            self.logger.error(f"Generation model for OpenAI was not set")
            return None
        
        # Use provided values or fall back to defaults
        max_output_tokens = max_output_tokens if max_output_tokens else self.default_output_max_tokens
        temperature = temperature if temperature else self.default_generation_temperature
        
        # Append current user prompt to chat history
        # Note: This modifies the original chat_history list passed in
        chat_history.append(self.construct_prompt(prompt, role=OpenAIEnums.USER.value))
        
        # Call OpenAI's chat completions API
        response = self.client.chat.completions.create(
            model=self.generation_model_id,     # Model to use for generation
            messages=chat_history,              # Full conversation history including current prompt
            max_tokens=max_output_tokens,       # Maximum tokens to generate
            temperature=temperature             # Randomness control
        )
        
        # Validate response structure and extract generated text
        # OpenAI returns: response.choices[0].message.content
        if not response or not response.choices or len(response.choices) == 0 or not response.choices[0].message:
            self.logger.error(f"Failed to get response from OpenAI for prompt")
            return None
        return response.choices[0].message.content

    def embed_text(self, text, document_type = None) -> list[float]:
        """
        Generate embedding vector for text using OpenAI's embeddings endpoint.
        
        Note: OpenAI does not differentiate between document and query embeddings
        (unlike Cohere), so document_type parameter is ignored.
        
        Args:
            text: Text to convert to embedding vector
            document_type: Ignored for OpenAI (no optimization difference)
            
        Returns:
            List of floats representing the embedding vector, or None if error
        """
        # Validate client is initialized
        if not self.client:
            self.logger.error(f"OpenAI Client was not set")
            return None
        
        # Validate embedding model is configured
        if not self.embedding_model_id:
            self.logger.error(f"Embedding model for OpenAI was not set")
            return None
        
        # Call OpenAI's embeddings API
        response = self.client.embeddings.create(
            input=self.process_text(text),
            model=self.embedding_model_id   # Embedding model to use
        )
        
        # Validate response structure and extract embedding vector
        # OpenAI returns: response.data[0].embedding
        if not response or not response.data or len(response.data) == 0 or not response.data[0].embedding:
            self.logger.error(f"Failed to get embedding from OpenAI for text: {text}")
            return None 
        
        return response.data[0].embedding  # Return the embedding vector
    
    def embed_batch(self, texts: list[str], document_type=None, batch_size: int = 100) -> list[list[float]]:
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                sub_batch = texts[i : i + batch_size]
                processed_texts = [self.process_text(t) for t in sub_batch]

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = self.client.embeddings.create(
                            input=processed_texts,
                            model=self.embedding_model_id
                        )
                        all_embeddings.extend([item.embedding for item in response.data])
                        break
                    except Exception as e:
                        if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                            time.sleep((2 ** attempt) + random.random())
                        else:
                            return None
            return all_embeddings
    
    def construct_prompt(self, prompt: str, role: str = "user") -> dict:
        """
        Format prompt as OpenAI-compatible message dictionary.
        
        Args:
            prompt: Raw prompt text
            role: Message role ("user", "system", or "assistant")
            
        Returns:
            Dictionary with role and processed content in OpenAI format
        """
        return {
            "role": role,                            # Message role (user/system/assistant)
            "content": self.process_text(prompt)     # Message content (truncated and cleaned)
        }
