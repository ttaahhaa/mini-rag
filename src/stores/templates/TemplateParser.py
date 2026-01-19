import os
import asyncio
import importlib
from models.enums.TemplatesEnum import TemplateDirectoriesAndFilesEnums

class TemplateParser():
    def __init__(self, default_language: str, language: str):
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.default_language = default_language
        self.language = None
        
        # Initial setup is sync because it happens at instantiation 
        # (usually during app startup or controller init)
        self._sync_set_language(language=language)

    def _sync_set_language(self, language: str):
        """Internal sync helper for initialization."""
        if not language:
            self.language = self.default_language
            return

        language_path = os.path.join(self.current_path, TemplateDirectoriesAndFilesEnums.LOCALES.value, language)
        if os.path.exists(language_path):
            self.language = language
        else:
            self.language = self.default_language

    async def get(self, group: str, key: str, vars: dict = {}):
        if not group or not key:
            return None

        targeted_language = await asyncio.to_thread(self._resolve_targeted_language, group)
        if not targeted_language:
            return None

        # This now imports the actual file (e.g., rag.py) as a module
        module = await asyncio.to_thread(self._import_template_module, targeted_language, group)

        # Check if the variable (key) exists directly in the module
        if not module or not hasattr(module, key):
            self.logger.warning(f"Template key '{key}' not found in {group}.py")
            return None

        key_attribute = getattr(module, key)
        
        # Ensure it's a Template object before substituting
        return key_attribute.substitute(vars)

    def _resolve_targeted_language(self, group: str) -> str:
        """Sync helper for path checking, meant to run in a thread."""
        group_filename = f"{group}.py"
        
        # Try current language
        path = os.path.join(self.current_path, TemplateDirectoriesAndFilesEnums.LOCALES.value, 
                            self.language, group_filename)
        if os.path.exists(path):
            return self.language
            
        # Fallback to default language
        default_path = os.path.join(self.current_path, TemplateDirectoriesAndFilesEnums.LOCALES.value, 
                                    self.default_language, group_filename)
        if os.path.exists(default_path):
            return self.default_language
            
        return None

    def _import_template_module(self, language: str, group: str):
        """Sync helper for module loading, meant to run in a thread."""
        try:
            # Construct path: stores.llm.templates.locales.{language}
            module_path = (
            f"{TemplateDirectoriesAndFilesEnums.STORES.value}."
            f"{TemplateDirectoriesAndFilesEnums.TEMPLATES.value}."
            f"{TemplateDirectoriesAndFilesEnums.LOCALES.value}."
            f"{language}."
            f"{group}"
        )
            return importlib.import_module(module_path)
        except ImportError:
            return None