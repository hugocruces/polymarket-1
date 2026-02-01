"""
LLM Provider Clients

Multi-provider LLM integration supporting Anthropic, OpenAI, and Google.
Each provider is abstracted behind a common interface for easy switching.

Configuration is done via environment variables:
- ANTHROPIC_API_KEY
- OPENAI_API_KEY  
- GOOGLE_API_KEY
"""

import asyncio
import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from polymarket_agent.config import LLMProvider, LLM_MODELS, get_api_key

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from an LLM call."""
    content: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""


class LLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    def __init__(self, model: str, api_key: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            model: Model identifier
            api_key: Optional API key (will use env var if not provided)
        """
        self.model = model
        self.api_key = api_key
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        pass
    
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            
        Returns:
            LLMResponse with generated content
        """
        pass
    
    def complete_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Synchronous wrapper for complete()."""
        return asyncio.run(self.complete(prompt, system_prompt, max_tokens, temperature))


class AnthropicClient(LLMClient):
    """
    Anthropic Claude API client.
    
    Supports Claude 3.5 Sonnet, Claude 3 Opus, and other Claude models.
    
    Requires: ANTHROPIC_API_KEY environment variable
    Docs: https://docs.anthropic.com/
    """
    
    def __init__(self, model: str = "claude-sonnet-4-5", api_key: Optional[str] = None):
        super().__init__(model, api_key)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client = None
    
    @property
    def provider_name(self) -> str:
        return "Anthropic"
    
    def _get_client(self):
        """Lazy initialization of the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
                self._client = AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: pip install anthropic"
                )
        return self._client
    
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate completion using Claude with streaming."""
        client = self._get_client()

        messages = [{"role": "user", "content": prompt}]

        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }

        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            # Use streaming to avoid timeout errors with the Anthropic SDK
            content = ""
            input_tokens = 0
            output_tokens = 0
            finish_reason = ""

            async with client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    content += text

                # Get final message for usage stats
                response = await stream.get_final_message()
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                finish_reason = response.stop_reason or ""

            return LLMResponse(
                content=content,
                model=self.model,
                provider=self.provider_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                finish_reason=finish_reason,
            )
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            raise


class OpenAIClient(LLMClient):
    """
    OpenAI API client.
    
    Supports GPT-4, GPT-4 Turbo, GPT-5, and other OpenAI models.
    
    Requires: OPENAI_API_KEY environment variable
    Docs: https://platform.openai.com/docs/
    """
    
    def __init__(self, model: str = "gpt-5.2", api_key: Optional[str] = None):
        super().__init__(model, api_key)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
    
    @property
    def provider_name(self) -> str:
        return "OpenAI"
    
    def _get_client(self):
        """Lazy initialization of the OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )
        return self._client
    
    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate completion using OpenAI."""
        client = self._get_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "temperature": temperature,
            }

            try:
                response = await client.chat.completions.create(**kwargs)
            except Exception as first_err:
                # Some models (o-series, mini reasoning) reject temperature
                if "temperature" in str(first_err).lower():
                    kwargs.pop("temperature", None)
                    response = await client.chat.completions.create(**kwargs)
                else:
                    raise

            choice = response.choices[0]

            return LLMResponse(
                content=choice.message.content or "",
                model=self.model,
                provider=self.provider_name,
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                output_tokens=response.usage.completion_tokens if response.usage else 0,
                finish_reason=choice.finish_reason or "",
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise


class GoogleClient(LLMClient):
    """
    Google GenAI (Gemini) client.

    Supports Gemini Pro, Gemini Flash, and other Google models.
    Uses the google-genai SDK with native async support.

    Requires: GOOGLE_API_KEY environment variable
    Docs: https://ai.google.dev/docs
    """

    def __init__(self, model: str = "gemini-2.5-flash", api_key: Optional[str] = None):
        super().__init__(model, api_key)
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._client = None

    @property
    def provider_name(self) -> str:
        return "Google"

    def _get_client(self):
        """Lazy initialization of the async Google GenAI client."""
        if self._client is None:
            try:
                from google import genai
                # .aio gives the native async client
                self._client = genai.Client(api_key=self.api_key).aio
            except ImportError:
                raise ImportError(
                    "google-genai package not installed. "
                    "Install with: pip install google-genai"
                )
        return self._client

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate completion using Gemini."""
        from google.genai import types

        client = self._get_client()

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        try:
            response = await client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

            usage = response.usage_metadata

            return LLMResponse(
                content=response.text,
                model=self.model,
                provider=self.provider_name,
                input_tokens=usage.prompt_token_count if usage else 0,
                output_tokens=usage.candidates_token_count if usage else 0,
                finish_reason="stop",
            )
        except Exception as e:
            logger.error(f"Google API error: {e}")
            raise


def get_llm_client(
    model: str = "claude-sonnet-4-5",
    api_key: Optional[str] = None,
) -> LLMClient:
    """
    Get an LLM client for the specified model.
    
    Automatically selects the correct provider based on the model name.
    
    Args:
        model: Model identifier (e.g., "claude-sonnet-4-5", "gpt-5.2", "gemini-2.5-flash")
        api_key: Optional API key (uses environment variable if not provided)
        
    Returns:
        LLMClient instance for the specified model
        
    Raises:
        ValueError: If model is not recognized
        
    Example:
        >>> client = get_llm_client("claude-sonnet-4-5")
        >>> response = await client.complete("Hello, world!")
    """
    if model not in LLM_MODELS:
        raise ValueError(
            f"Unknown model: {model}. "
            f"Available models: {', '.join(LLM_MODELS.keys())}"
        )
    
    model_config = LLM_MODELS[model]
    provider = model_config["provider"]
    model_id = model_config["model_id"]
    
    if provider == LLMProvider.ANTHROPIC:
        return AnthropicClient(model=model_id, api_key=api_key)
    elif provider == LLMProvider.OPENAI:
        return OpenAIClient(model=model_id, api_key=api_key)
    elif provider == LLMProvider.GOOGLE:
        return GoogleClient(model=model_id, api_key=api_key)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def validate_llm_setup(model: str) -> tuple[bool, str]:
    """
    Validate that the LLM setup is correct for the given model.
    
    Checks for required API keys and installed packages.
    
    Args:
        model: Model identifier
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if model not in LLM_MODELS:
        return False, f"Unknown model: {model}"
    
    model_config = LLM_MODELS[model]
    provider = model_config["provider"]
    
    # Check API key
    api_key = get_api_key(provider)
    if not api_key:
        key_names = {
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.GOOGLE: "GOOGLE_API_KEY",
        }
        return False, f"Missing API key: {key_names[provider]}"
    
    # Check package installation
    try:
        if provider == LLMProvider.ANTHROPIC:
            import anthropic
        elif provider == LLMProvider.OPENAI:
            import openai
        elif provider == LLMProvider.GOOGLE:
            from google import genai
    except ImportError as e:
        return False, f"Missing package: {e}"
    
    return True, ""
