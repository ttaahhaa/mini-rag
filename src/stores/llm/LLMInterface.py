from abc import ABC, abstractmethod
from typing import List, Optional

class LLMInterface(ABC):
    """
    Abstract Base Class defining the contract for all LLM provider implementations.
    Forces concrete classes (CohereProvider, OpenAIProvider, etc.) to implement
    standardized methods for text generation and embedding operations.
    
    This interface enables the Factory Pattern in LLMProviderFactory, allowing
    the application to switch between different LLM providers without changing
    the business logic code.
    """
    
    @abstractmethod
    def set_generation_model(self, model_id: str):
        """
        Configure which model to use for text generation tasks.
        
        Called during application startup (in lifespan function) to set the
        generation model based on environment configuration (GENERATION_MODEL_ID).
        
        Args:
            model_id: Provider-specific model identifier
                     Examples: "command-r-plus" (Cohere), "gpt-4" (OpenAI)
        
        Implementation requirements:
            - Store the model_id for use in generate_text()
            - Log the configuration change
            - Validate model availability (optional)
        """
        pass

    @abstractmethod
    def set_embedding_model(self, model_id: str, embedding_size: int):
        """
        Configure which model to use for generating text embeddings.
        
        Called during application startup to set the embedding model based on
        environment configuration (EMBEDDING_MODEL_ID, EMBEDDING_MODEL_SIZE).
        
        Args:
            model_id: Provider-specific embedding model identifier
                     Examples: "embed-english-v3.0" (Cohere), "text-embedding-3-small" (OpenAI)
            embedding_size: Dimension of the embedding vectors (e.g., 768, 1024, 1536)
                           Must match the model's output dimension and MongoDB vector index
        
        Implementation requirements:
            - Store model_id and embedding_size for use in embed_text()
            - Ensure embedding_size matches the model's actual output
            - Log the configuration change
        """
        pass

    @abstractmethod
    def generate_text(self, prompt: str, chat_history: list=[], max_output_tokens: int = None, 
                     temperature: float = None) -> str:
        """
        Generate text response from the LLM based on a prompt and optional chat history.
        
        This is the core method for RAG generation - takes retrieved context and user query,
        sends to LLM, and returns the generated answer.
        
        Args:
            prompt: The user's question or instruction to generate text for
                   In RAG: typically includes both retrieved context and user query
            chat_history: List of previous conversation messages for context
                         Format varies by provider (e.g., [{"role": "user", "content": "..."}])
                         Default: empty list (no history)
            max_output_tokens: Maximum tokens in generated response (overrides default)
                              Controls response length and API costs
                              Default: None (uses provider's default setting)
            temperature: Sampling temperature controlling randomness (0.0 to 1.0+)
                        0.0 = deterministic, higher = more creative/random
                        Default: None (uses provider's default, typically 0.1)
        
        Returns:
            Generated text response as string
            Returns None if generation fails (client not initialized, model not set, API error)
        
        Implementation requirements:
            - Validate client and model are configured
            - Preprocess prompt (truncation, formatting)
            - Call provider's chat/completion API
            - Handle errors gracefully with logging
            - Return plain text response
        """
        pass

    def embed_text(self, text: str, document_type: Optional[str] = None) -> List[float]:
        """Convert a single string into a vector (used for queries)."""
        pass
        """
        Convert text into a dense vector embedding for semantic search.
        
        This is the core method for RAG retrieval - embeds both documents (for indexing)
        and queries (for searching). The embeddings enable finding semantically similar
        documents in the vector database (MongoDB with vector search).
        
        Args:
            text: Text to convert into embedding vector
                 For documents: chunk of text from uploaded files
                 For queries: user's search question
            document_type: Optimization hint for the embedding model
                          LLMEnums.QUERY: text is a search query (optimize for search)
                          None/other: text is a document (optimize for indexing)
                          
                          Some providers (like Cohere) use this to optimize embeddings
                          differently for queries vs documents, improving retrieval accuracy.
        
        Returns:
            List of floats representing the embedding vector
            Length matches embedding_size set in set_embedding_model()
            Returns None if embedding fails (client not initialized, model not set, API error)
        
        Implementation requirements:
            - Validate client and embedding model are configured
            - Preprocess text (truncation, cleaning)
            - Handle document_type appropriately for provider
            - Call provider's embedding API
            - Extract and return float vector
            - Handle errors gracefully with logging
        
        Usage in RAG:
            - Index time: embed_text(document_chunk, document_type=None) → store in MongoDB
            - Query time: embed_text(user_query, document_type=LLMEnums.QUERY) → search MongoDB
        """
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str], document_type: Optional[str] = None) -> List[List[float]]:
        """
        Convert a list of strings into a list of vectors in a single call.
        
        This is the high-performance method for RAG indexing. 
        Implementations should include sub-batching and retry logic.
        """
        pass

    @abstractmethod
    def construct_prompt(self, prompt: str, role: str = "user") -> dict:
        """
        Format a text prompt into provider-specific message structure.
        
        Different LLM providers expect different message formats. This method
        standardizes the formatting so the rest of the application doesn't need
        to know provider-specific details.
        
        Args:
            prompt: Raw text content of the message
            role: The speaker/role of the message
                 Common values: "user" (end user), "system" (instructions), "assistant" (AI)
                 Note: Role names may differ by provider (e.g., Cohere uses "chatbot" not "assistant")
        
        Returns:
            Dictionary containing the formatted message
            Common structure: {"role": "user", "content": "processed text"}
            
        Implementation requirements:
            - Preprocess the prompt text (via process_text() or similar)
            - Map role to provider-specific role names if needed
            - Return provider-compatible message dictionary
            
        Provider-specific examples:
            OpenAI: {"role": "user", "content": "What is Python?"}
            Cohere: {"role": "user", "content": "What is Python?"}
                   (but "assistant" → "chatbot" for chat history)
        """
        pass
