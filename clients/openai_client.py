"""
OpenAI API client for interacting with OpenAI models.

This module provides a client for making requests to OpenAI API for text generation
and model management.
"""

import httpx
import traceback
import logging
from typing import List, Dict, Any, Optional, ClassVar

# Use standard Python logging
logger = logging.getLogger(__name__)

class OpenAIClient:
    """Client for interacting with OpenAI API."""

    _instance: ClassVar[Optional['OpenAIClient']] = None

    @classmethod
    def get_instance(cls) -> 'OpenAIClient':
        """Get singleton instance of OpenAIClient, ensuring reuse of the same client."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, api_base: str = None, api_key: str = None, model: str = None,
                 max_tokens: int = 4096, temperature: float = 0.2):
        """
        Initialize OpenAI API client.

        Args:
            api_base: OpenAI API base URL
            api_key: OpenAI API key
            model: OpenAI model to use
            max_tokens: Maximum tokens for completion
            temperature: Temperature for generation
        """
        # Import locally to avoid circular imports
        from llm_config import load_llm_config

        # Load config
        config = load_llm_config()
        active_provider = config.get("active_llm_provider", "ollama")
        if active_provider == "openai" and "llm_configs" in config and "openai" in config["llm_configs"]:
            openai_config = config["llm_configs"]["openai"]
            self.api_base = api_base or openai_config.get("llm_api_base", "https://api.openai.com/v1")
            self.api_key = api_key or openai_config.get("llm_api_key", "")
            self.model = model or openai_config.get("llm_model", "gpt-3.5-turbo")
            self.max_tokens = max_tokens or int(openai_config.get("llm_max_tokens", 4096))
            self.temperature = temperature or float(openai_config.get("llm_temperature", 0.2))
        else:
            self.api_base = api_base or "https://api.openai.com/v1"
            self.api_key = api_key or ""
            self.model = model or "gpt-3.5-turbo"
            self.max_tokens = max_tokens
            self.temperature = temperature

        # Get timeout from config or use default
        timeout = config.get("llm_configs", {}).get("openai", {}).get("llm_request_timeout", 60)
        self.client = httpx.Client(timeout=float(timeout))

        # Validate API key
        if not self.api_key:
            logger.warning("OpenAI API key not provided")

    def generate(self, prompt: str) -> str:
        """Generate a response using the OpenAI model."""
        if not self.api_key:
            error_msg = "OpenAI API key not provided. Please set it in config.json."
            logger.error(error_msg)
            return f"Error: {error_msg}"

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }

            # Log API call
            logger.info(f"Calling OpenAI API with model: {self.model}")
            logger.debug(f"Prompt length: {len(prompt)} characters")

            response = self.client.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            response_data = response.json()

            # Extract the assistant's message content
            if "choices" in response_data and len(response_data["choices"]) > 0:
                content = response_data["choices"][0]["message"]["content"]
                logger.info(f"OpenAI response received, length: {len(content)} characters")
                return content

            error_msg = "Could not extract content from OpenAI response"
            logger.error(error_msg)
            return error_msg

        except httpx.HTTPStatusError as e:
            error_message = f"OpenAI API error (HTTP {e.response.status_code}): "
            try:
                error_data = e.response.json()
                if "error" in error_data:
                    error_message += error_data["error"].get("message", str(e))
                else:
                    error_message += str(e)
            except Exception:
                error_message += str(e)

            logger.error(error_message, exc_info=True)
            return f"Error generating response: {error_message}"

        except Exception as e:
            error_message = f"Error calling OpenAI API: {e}"
            logger.error(error_message, exc_info=True)
            return f"Error generating response: {error_message}"

    def get_models(self) -> List[Dict[str, Any]]:
        """List available models in OpenAI."""
        if not self.api_key:
            logger.warning("OpenAI API key not provided. Cannot list models.")
            return []

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}"
            }

            response = self.client.get(
                f"{self.api_base}/models",
                headers=headers
            )
            response.raise_for_status()

            # Extract model list
            models_data = response.json().get("data", [])

            # Log available models
            model_ids = [m.get('id') for m in models_data]
            logger.info(f"Available OpenAI models: {model_ids}")

            return models_data

        except Exception as e:
            error_message = f"Error listing OpenAI models: {e}"
            logger.error(error_message, exc_info=True)
            return []
