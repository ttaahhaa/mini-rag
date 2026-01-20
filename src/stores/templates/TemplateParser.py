import os
import asyncio
import importlib
import logging
from typing import Optional, Any
from string import Template, _TemplateMetaclass
from models.enums.TemplatesEnum import TemplateDirectoriesAndFilesEnums

class TemplateParser:
    def __init__(self, default_language: str, language: str):
        self.logger = logging.getLogger(__name__)
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.default_language = default_language
        self.language = None
        self._module_cache = {}
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
            self.logger.warning(f"Language '{language}' not found, falling back to default '{self.default_language}'")
            self.language = self.default_language
            
class TemplateNotFound(Exception):
    """Custom exception for when a template is not found."""

class InvalidTemplate(Exception):
    """Custom exception for when a template is not a Template object."""

    async def get(self, group: str, key: str, vars: dict = {}) -> Optional[str]:
        if not group or not key:
            return None

        targeted_language = await asyncio.to_thread(self._resolve_targeted_language, group)
        if not targeted_language:
            self.logger.warning(f"Could not resolve language for group '{group}'")
            return None

        # This now imports the actual file (e.g., rag.py) as a module
        module = await asyncio.to_thread(self._import_template_module, targeted_language, group)

        if module is None:
            raise TemplateNotFound(f"Could not import template module '{group}' for language '{targeted_language}'")

        # Check if the variable (key) exists directly in the module
        if not module or not hasattr(module, key):
            self.logger.warning(f"Template key '{key}' not found in {group}.py ({targeted_language})")
            raise TemplateNotFound(f"Template key '{key}' not found in {group}.py ({targeted_language})")

        key_attribute = getattr(module, key)
        
        # Ensure it's a Template object before substituting
        if isinstance(key_attribute, Template):
            return key_attribute.substitute(vars)
        elif isinstance(key_attribute, str):
            return key_attribute
        else:
            raise InvalidTemplate(f"Attribute '{key}' in {group}.py is not a Template object")

    def _resolve_targeted_language(self, group: str) -> Optional[str]:
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

    def _import_template_module(self, language: str, group: str) -> Any:
        """Sync helper for module loading, meant to run in a thread."""
        try:
            # Check cache first
            if (language, group) in self._module_cache:
                return self._module_cache[(language, group)]

            # Construct path: stores.templates.locales.{language}.{group}
            module_path = (
                f"{TemplateDirectoriesAndFilesEnums.STORES.value}."
                f"{TemplateDirectoriesAndFilesEnums.TEMPLATES.value}."
                f"{TemplateDirectoriesAndFilesEnums.LOCALES.value}."
                f"{language}."
                f"{group}"
            )
            
            module = importlib.import_module(module_path)

            # Cache the module
            self._module_cache[(language, group)] = module
            return module
        except ImportError as e:
            self.logger.error(f"Failed to import template module '{group}' for language '{language}': {e}")
            raise
        
