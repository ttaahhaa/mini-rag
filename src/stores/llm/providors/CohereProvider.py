from ..LLMInterface import LLMInterface
from ..LLMEnums import CohereEnums, LLMEnums
from helpers.config import get_settings
import cohere
import logging


class CohereProvider(LLMInterface):  # FIXED: Corrected spelling from "Providor" to "Provider"
    """
    Concrete implementation of LLMInterface for Cohere API integration.
    Provides text generation and embedding capabilities using Cohere's models.
    """
    
    def __init__(self, api_key: str = None, default_input_max_tokens: int = 1000,
                 default_output_max_tokens: int = 1000,
                 temperature: float = 0.1):
        settings = get_settings()
        self.api_key = api_key or settings.COHERE_API_KEY
        
        # Maximum tokens to process from input text (truncation limit)
        self.default_input_max_tokens = default_input_max_tokens
        
        # Maximum tokens the model can generate in response
        self.default_output_max_tokens = default_output_max_tokens  
        
        # Controls randomness in generation (0.0 = deterministic, higher = more creative)
        self.default_generation_temperature = temperature
        
        # Model identifiers - set later via setter methods
        self.generation_model_id = None  # e.g., "command-r-plus"
        self.embedding_model_id = None   # e.g., "embed-english-v3.0"
        self.embedding_size = None   
        
        # Initialize Cohere client with API key
        self.client = cohere.Client(self.api_key)
        
        # Logger for tracking operations and debugging
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        """
        Configure which Cohere model to use for text generation.
        Called during app startup via LLMProviderFactory.
        
        Args:
            model_id: Cohere model identifier (e.g., "command-r-plus")
        """
        self.generation_model_id = model_id
        self.logger.info(f"Set generation model to {model_id}")
    
    def set_embedding_model(self, model_id: str, embedding_size: int):
        """
        Configure which Cohere model to use for embeddings.
        
        Args:
            model_id: Cohere embedding model (e.g., "embed-english-v3.0")
            embedding_size: Vector dimensions (must match model's output size)
        """
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size
        self.logger.info(f"Set embedding model to {model_id} with size {embedding_size}")
    
    def process_text(self, text: str) -> str:
        """
        Preprocess text before sending to Cohere API.
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
        Generate text using Cohere's chat endpoint.
        
        Args:
            prompt: User message/question to generate response for
            chat_history: List of previous messages for context (Cohere chat format)
            max_output_tokens: Override default max output tokens
            temperature: Override default temperature for this generation
            
        Returns:
            Generated text response or None if error occurs
        """
        # Validate client is initialized
        if not self.client:
            self.logger.error(f"Cohere Client was not set") 
            return None
        
        # Validate generation model is configured
        if not self.generation_model_id:
            self.logger.error(f"Generation model for Cohere was not set")
            return None
        
        # Call Cohere's chat API with configuration
        response = self.client.chat(
            model=self.generation_model_id,                     # Model to use for generation
            chat_history=chat_history,                          # Previous conversation context
            message=self.construct_prompt(prompt),              # Current user message (formatted as dict)
            max_tokens=max_output_tokens if max_output_tokens else self.default_output_max_tokens,
            temperature=temperature if temperature else self.default_generation_temperature          # Randomness control
        )
        
        # Validate response and extract generated text
        if not response or not response.text:
            self.logger.error(f"Failed to get response from Cohere for prompt")
            return None
        return response.text

    def embed_text(self, text, document_type = None) -> list[float]:
        """
        Generate embedding vector for text using Cohere's embed endpoint.
        Cohere optimizes embeddings differently for documents vs queries.
        
        Args:
            text: Text to convert to embedding vector
            document_type: LLMEnums.QUERY for search queries, otherwise treated as document
            
        Returns:
            List of floats representing the embedding vector, or None if error
        """
        # Validate client is initialized
        if not self.client:
            self.logger.error(f"Cohere Client was not set")
            return None
        
        # Validate embedding model is configured
        if not self.embedding_model_id:
            self.logger.error(f"Embedding model for Cohere was not set")
            return None
        
        # Determine input type - Cohere uses different optimization for documents vs queries
        input_type = CohereEnums.DOCUMNET  # Default: document for indexing
        if document_type == LLMEnums.QUERY:
            input_type = CohereEnums.QUERY  # Use query optimization for search
        
        # Call Cohere's embed API
        response = self.client.embed(
            model=self.embedding_model_id,           # Embedding model to use
            texts=[self.process_text(text)],         # Text to embed (as list, truncated to max tokens)
            input_type=input_type,                   # "search_document" or "search_query"
            embedding_types=["float"]                # Return format (float32 vectors)
        )
        
        # FIXED: Corrected logic - added "not" before response.embeddings.float
        if not response or not response.embeddings or not response.embeddings.float:
            self.logger.error(f"Failed to get embeddings from Cohere for text")
            return None
        return response.embeddings.float[0]  # Return first (and only) embedding

    def construct_prompt(self, prompt: str, role: str = "user") -> dict:
        """
        Format prompt as Cohere-compatible message dictionary.
        
        Args:
            prompt: Raw prompt text
            role: Message role ("user" for user messages, "system" for system prompts)
            
        Returns:
            Dictionary with role and processed content
        """
        return {
            "role": role,                            # Who is speaking (user/system)
            "content": self.process_text(prompt)     # Message content (truncated and cleaned)
        }
