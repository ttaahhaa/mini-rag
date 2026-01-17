from ..llm.providers import CohereProvider, OpenAIProvider
from .LLMEnums import LLMEnums
from helpers.config import Settings

class LLMProviderFactory:
    def __init__(self, config: Settings = None):
        self.config = config

    def create(self, provider_name: str):
        if provider_name == LLMEnums.OPENAI.value:
            return OpenAIProvider(
                api_key=self.config.OPENAI_API_KEY,
                api_url=self.config.OPENAI_API_URL,
                default_input_max_tokens=self.config.DEFAULT_INPUT_MAX_TOKENS,
                defualt_output_max_tokens=self.config.DEFUALT_OUTPUT_MAX_TOKENS,
                temperature=self.config.DEFAULT_GENERATION_TEMPERATURE
            )
        elif provider_name == LLMEnums.COHERE.value:
            return CohereProvider(
                api_key=self.config.COHERE_API_KEY,
                default_input_max_tokens=self.config.DEFAULT_INPUT_MAX_TOKENS,
                default_output_max_tokens=self.config.DEFUALT_OUTPUT_MAX_TOKENS,
                temperature=self.config.DEFAULT_GENERATION_TEMPERATURE
            )
         
        else:
            raise ValueError(f"Unsupported provider: {provider_name}")