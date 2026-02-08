"""
LLM API client package for C# code review agent.

This package provides clients for interfacing with different LLM providers.
"""

from .ollama_client import OllamaClient
from .openai_client import OpenAIClient

# Factory function to get the appropriate client based on configuration
def get_llm_client():
    """
    Get the appropriate LLM client based on configuration.

    Returns:
        An instance of the appropriate LLM client
    """
    # Import locally to avoid circular imports
    from llm_config import load_llm_config

    config = load_llm_config()
    active_provider = config.get("active_llm_provider", "ollama")

    if active_provider == "openai":
        return OpenAIClient.get_instance()
    else:
        return OllamaClient.get_instance()
