import os
import asyncio
import importlib
import logging
from typing import Optional, Any
from string import Template
from models.enums.TemplatesEnum import TemplateDirectoriesAndFilesEnums


class TemplateParser:
    """
    A utility class to parse and manage string templates stored in Python modules.
    Supports multi-language locales with a fallback mechanism to a default language.
    """
    def __init__(self, default_language: str, language: str):
        """
        Initializes the TemplateParser.

        Args:
            default_language (str): The fallback language code (e.g., 'en').
            language (str): The preferred language code.
        """
        self.logger = logging.getLogger(__name__)
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.default_language = default_language
        self.language = None
        # Cache to store imported modules to avoid redundant I/O and import overhead
        self._module_cache = {}
        self._sync_set_language(language=language)

    def _sync_set_language(self, language: str):
        """
        Synchronously validates and sets the preferred language.
        Falls back to default_language if the specified language locale directory doesn't exist.
        """
        if not language:
            self.language = self.default_language
            return

        language_path = os.path.join(
            self.current_path,
            TemplateDirectoriesAndFilesEnums.LOCALES.value,
            language,
        )
        if os.path.exists(language_path):
            self.language = language
        else:
            self.logger.warning(
                f"Language '{language}' not found, falling back to default '{self.default_language}'"
            )
            self.language = self.default_language

    async def get(self, group: str, key: str, vars: dict = {}) -> Optional[str]:
        if not group or not key:
            return None

        targeted_language = await asyncio.to_thread(
            self._resolve_targeted_language, group
        )
        if not targeted_language:
            self.logger.warning(
                f"Could not resolve language for group '{group}'"
            )
            return None

        module = await asyncio.to_thread(
            self._import_template_module, targeted_language, group
        )
        if module is None:
            raise TemplateNotFound(
                f"Could not import template module '{group}' for language '{targeted_language}'"
            )

        if not hasattr(module, key):
            self.logger.warning(
                f"Template key '{key}' not found in {group}.py ({targeted_language})"
            )
            raise TemplateNotFound(
                f"Template key '{key}' not found in {group}.py ({targeted_language})"
            )

        key_attribute = getattr(module, key)

        # Handle string.Template objects for dynamic substitution
        if isinstance(key_attribute, Template):
            return key_attribute.substitute(vars)
        # Handle plain strings
        elif isinstance(key_attribute, str):
            return key_attribute
        else:
            raise InvalidTemplate(
                f"Attribute '{key}' in {group}.py is not a Template object"
            )

    def _resolve_targeted_language(self, group: str) -> Optional[str]:
        """
        Determines which language locale contains the requested template group.
        Checks preferred language first, then falls back to default.
        """
        group_filename = f"{group}.py"

        path = os.path.join(
            self.current_path,
            TemplateDirectoriesAndFilesEnums.LOCALES.value,
            self.language,
            group_filename,
        )
        if os.path.exists(path):
            return self.language

        default_path = os.path.join(
            self.current_path,
            TemplateDirectoriesAndFilesEnums.LOCALES.value,
            self.default_language,
            group_filename,
        )
        if os.path.exists(default_path):
            return self.default_language

        return None

    def _import_template_module(self, language: str, group: str) -> Any:
        """
        Dynamically imports the template module using importlib.
        Uses a local cache to improve performance on subsequent requests.
        """
        try:
            # Check if the module is already in the cache
            if (language, group) in self._module_cache:
                return self._module_cache[(language, group)]

            # Construct the full module path for importlib
            module_path = (
                f"{TemplateDirectoriesAndFilesEnums.STORES.value}."
                f"{TemplateDirectoriesAndFilesEnums.TEMPLATES.value}."
                f"{TemplateDirectoriesAndFilesEnums.LOCALES.value}."
                f"{language}."
                f"{group}"
            )

            module = importlib.import_module(module_path)
            # Store in cache for future use
            self._module_cache[(language, group)] = module
            return module
        except ImportError as e:
            self.logger.error(
                f"Failed to import template module '{group}' for language '{language}': {e}"
            )
            raise


class TemplateNotFound(Exception):
    """Custom exception for when a template is not found."""


class InvalidTemplate(Exception):
    """Custom exception for when a template is not a Template object."""
