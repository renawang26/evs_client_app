"""
Ollama API client for interacting with Ollama models.

This module provides a client for making requests to Ollama API for text generation
and model management.
"""

import json
import httpx
import traceback
import logging
from typing import List, Dict, Any, Optional, ClassVar

# Use standard Python logging
logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API."""

    _instance: ClassVar[Optional['OllamaClient']] = None

    @classmethod
    def get_instance(cls) -> 'OllamaClient':
        """Get singleton instance of OllamaClient, ensuring reuse of the same client."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self, base_url: str = None, model: str = None):
        """
        Initialize Ollama API client.

        Args:
            base_url: Ollama API base URL
            model: Ollama model to use
        """
        # Import locally to avoid circular imports
        from llm_config import load_llm_config

        # Load config
        config = load_llm_config()
        self.base_url = base_url or config.get("llm_configs", {}).get("ollama", {}).get("llm_base_url", "http://localhost:11434")
        self.model = model or config.get("llm_configs", {}).get("ollama", {}).get("llm_model", "phi3")

        # Get timeout from config or use default
        timeout = float(config.get("llm_configs", {}).get("ollama", {}).get("llm_request_timeout", 60))
        self.client = httpx.Client(timeout=timeout)

    def generate(self, prompt: str) -> str:
        """Generate a response using the Ollama model."""
        try:
            # Log API call
            logger.info(f"Calling Ollama API with model: {self.model}")
            logger.debug(f"Prompt length: {len(prompt)} characters")

            # First try using /api/generate
            response = self.client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            content = response.json().get("response", "")

            # Log response
            logger.info(f"Ollama response received, length: {len(content)} characters")
            return content

        except httpx.HTTPStatusError as e:
            # If /api/generate fails, try using /api/chat
            if e.response.status_code == 404:
                try:
                    logger.info("Falling back to /api/chat endpoint")
                    chat_response = self.client.post(
                        f"{self.base_url}/api/chat",
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "stream": False
                        }
                    )
                    chat_response.raise_for_status()
                    # Need to process the entire response stream
                    result = chat_response.text
                    try:
                        # Try to parse JSON response
                        last_response = None
                        for line in result.splitlines():
                            if line.strip():
                                json_resp = json.loads(line)
                                if json_resp.get("done", False):
                                    last_response = json_resp

                        if last_response and "message" in last_response:
                            content = last_response["message"].get("content", "")
                            logger.info(f"Ollama chat response received, length: {len(content)} characters")
                            return content

                        error_msg = "Could not extract content from Ollama response"
                        logger.error(error_msg)
                        return error_msg

                    except json.JSONDecodeError:
                        logger.warning("Failed to parse JSON response, returning raw result")
                        return result

                except httpx.ConnectError:
                    error_msg = (
                        f"Cannot connect to Ollama at {self.base_url}. "
                        "Please make sure Ollama is running (start it with 'ollama serve')."
                    )
                    logger.error(error_msg)
                    raise ConnectionError(error_msg)
                except httpx.HTTPStatusError as chat_err:
                    error_msg = (
                        f"Ollama API not available at {self.base_url} "
                        f"(HTTP {chat_err.response.status_code}). "
                        "Please check that Ollama is running and the model is available "
                        "(run 'ollama list' to verify)."
                    )
                    logger.error(error_msg)
                    raise ConnectionError(error_msg)
                except Exception as chat_err:
                    error_msg = f"Error using chat API: {chat_err}"
                    logger.error(error_msg, exc_info=True)
                    return error_msg

            error_msg = f"Error generating response: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

        except httpx.ConnectError:
            error_msg = (
                f"Cannot connect to Ollama at {self.base_url}. "
                "Please make sure Ollama is running (start it with 'ollama serve')."
            )
            logger.error(error_msg)
            raise ConnectionError(error_msg)

        except Exception as e:
            error_msg = f"Error calling Ollama API: {e}"
            logger.error(error_msg, exc_info=True)
            return error_msg

    def get_models(self) -> List[Dict[str, Any]]:
        """List available models in Ollama."""
        try:
            # Use the newer API tags endpoint
            response = self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            # Extract model list
            models_data = response.json().get("models", [])

            # Log available models
            model_names = [m.get('name') for m in models_data]
            logger.info(f"Available Ollama models: {model_names}")

            return models_data
        except Exception as e:
            error_msg = f"Error listing Ollama models: {e}"
            logger.error(error_msg, exc_info=True)
            return []
