from .providers import CohereProvider, OpenAIProvider
from .LLMEnums import LLMEnums
from helpers.config import Settings

class LLMProviderFactory:
    def __init__(self, config: Settings = None):
        self.config = config

    def create(self, provider_name: str):
        """
        Instantiates LLM providers. 
        Ensure these providers use AsyncClients internally (e.g., OpenAI(AsyncClient)).
        """
        common_params = {
            "api_key": self.config.OPENAI_API_KEY if provider_name == LLMEnums.OPENAI.value else self.config.COHERE_API_KEY,
            "default_input_max_tokens": self.config.DEFAULT_INPUT_MAX_TOKENS,
            "default_output_max_tokens": self.config.DEFUALT_OUTPUT_MAX_TOKENS,
            "temperature": self.config.DEFAULT_GENERATION_TEMPERATURE
        }

        if provider_name == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                **common_params,
                api_url=self.config.OPENAI_API_URL
            )
        
        elif provider_name == LLMEnums.COHERE.value:
            return CohereProvider(**common_params)
         
        else:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")