from ..LLMInterface import LLMInterface
from ..LLMEnums import CohereEnums, DocumentTypeEnum, LLMEnums
from helpers.config import get_settings
import cohere
import logging
import time
import random


class CohereProvider(LLMInterface):
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
        self.enums = CohereEnums
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

    def generate_text(self, prompt: str, chat_history: list = [], max_output_tokens: int = None,
                  temperature: float = None) -> str:
        """
        Generate text using Cohere's chat endpoint.
        Fixed to ensure 'message' is a string to avoid 422 errors.
        """
        if not self.client or not self.generation_model_id:
            self.logger.error("Cohere Client or Model ID not configured.")
            return None
        
        # 1. Cohere's 'message' parameter MUST be a string.
        # We extract just the text, not the dictionary from construct_prompt.
        message_text = prompt
        if isinstance(prompt, dict):
            message_text = prompt.get("content", prompt.get("message", str(prompt)))
        else:
            # Just in case, process the raw string for truncation/cleaning
            message_text = self.process_text(prompt)

        try:
            response = self.client.chat(
                model=self.generation_model_id,
                chat_history=chat_history, 
                message=message_text, # This is a string
                max_tokens=max_output_tokens if max_output_tokens else self.default_output_max_tokens,
                temperature=temperature if temperature else self.default_generation_temperature
            )
            
            if not response or not response.text:
                return None
                
            return response.text

        except Exception as e:
            self.logger.error(f"Cohere API Error: {e}")
            return None

    def embed_text(self, text, document_type=None) -> list[float]:
        """
        Generate embedding vector for text using Cohere's embed endpoint.
        """
        # 1. Validation Logic
        if not self.client:
            self.logger.error(f"Cohere Client was not set")
            return None
        
        if not self.embedding_model_id:
            self.logger.error(f"Embedding model for Cohere was not set")
            return None

        # 2. Map internal types to Cohere-specific strings
        # Cohere V3+ requires: "search_document", "search_query", etc.
        cohere_input_type = CohereEnums.DOCUMENT.value # Default: "search_document"
        
        if document_type == DocumentTypeEnum.QUERY:
            cohere_input_type = CohereEnums.QUERY.value # "search_query"
        
        # 3. Call Cohere's embed API
        try:
            response = self.client.embed(
                model=self.embedding_model_id,
                texts=[self.process_text(text)],
                input_type=cohere_input_type, # Passing the string value
                embedding_types=["float"]
            )
            
            # 4. Check response and return first (and only) embedding
            if not response or not response.embeddings or not response.embeddings.float:
                self.logger.error(f"Failed to get embeddings from Cohere for text")
                return None
                
            return response.embeddings.float[0]
            
        except Exception as e:
            self.logger.error(f"Error during Cohere embedding: {str(e)}")
            return None

    def embed_batch(self, texts: list[str], document_type=None, batch_size: int = 96) -> list[list[float]]:
        """
        Converts a list of strings into a list of embedding vectors.
        
        This method is optimized for high-volume indexing by:
        1. Splitting the input into 'sub-batches' to avoid API payload limits.
        2. Implementing exponential backoff to handle 'Too Many Requests' (429) errors.
        """
        
        # Determine the purpose of the embedding (Indexing vs. Searching).
        # Cohere V3 models require an input_type to optimize vector performance.
        cohere_input_type = CohereEnums.DOCUMENT.value # Default: for storing in Vector DB
        if document_type == DocumentTypeEnum.QUERY:
            cohere_input_type = CohereEnums.QUERY.value # For real-time search queries

        all_embeddings = []
        
        # --- SUB-BATCHING LOOP ---
        # We iterate through the list of texts in steps of 'batch_size'.
        # This prevents sending requests that are physically too large for the API to process.
        for i in range(0, len(texts), batch_size):
            # Extract a slice of the text list (e.g., from index 0 to 95)
            sub_batch = texts[i : i + batch_size]
            
            # Pre-process each text in the sub-batch (truncation, cleaning, etc.)
            processed_texts = [self.process_text(t) for t in sub_batch]
            
            # --- RETRY LOGIC (EXPONENTIAL BACKOFF) ---
            # If the API returns a rate limit error (429), we wait and try again.
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Execute the API call for the current sub-batch
                    response = self.client.embed(
                        model=self.embedding_model_id,
                        texts=processed_texts,
                        input_type=cohere_input_type,
                        embedding_types=["float"]
                    )
                    
                    # If successful, add these vectors to our final list and exit the retry loop
                    all_embeddings.extend(response.embeddings.float)
                    break 
                    
                except Exception as e:
                    # Check if the error is specifically a rate limit (HTTP 429)
                    if "429" in str(e) and attempt < max_retries - 1:
                        # Wait logic: 2^attempt gives us 1s, 2s, 4s...
                        # random.random() adds 'jitter' to prevent simultaneous retries from multiple users.
                        wait_time = (2 ** attempt) + random.random()
                        self.logger.warning(f"Rate limit hit. Retrying in {wait_time:.2f}s...")
                        time.sleep(wait_time)
                    else:
                        # If the error isn't a 429, or we ran out of retries, we log it and fail.
                        self.logger.error(f"Batch failed after {max_retries} attempts: {e}")
                        return None
                        
        return all_embeddings
    
    def construct_prompt(self, prompt: str, role: str = "USER") -> dict:
        """
        Format prompt for Cohere-compatible chat history.
        Cohere requires 'message' instead of 'content'.
        """
        # Standardize roles for Cohere: USER or CHATBOT
        cohere_role = "USER" if role.lower() == "user" else "CHATBOT"
        
        return {
            "role": cohere_role,
            "message": self.process_text(prompt)
        }
