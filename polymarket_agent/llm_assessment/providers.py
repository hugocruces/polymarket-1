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
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, TypeVar

from polymarket_agent.config import LLM_MODELS, LLMProvider, get_api_key

logger = logging.getLogger(__name__)


# Retry configuration for transient LLM errors.
DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_BACKOFF = 1.0  # seconds
DEFAULT_MAX_BACKOFF = 30.0  # seconds


T = TypeVar("T")


def _is_transient(exc: BaseException) -> bool:
    """Heuristically decide whether an exception is worth retrying.

    Provider SDKs expose different exception types but share common patterns:
    rate limits, connection errors, timeouts, and 5xx responses are retriable;
    auth, bad-request, and not-found errors are not.
    """
    # Timeouts and cancellation-adjacent errors from asyncio or socket.
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, ConnectionError)):
        return True

    # Provider SDKs name their classes consistently — match without importing.
    name = type(exc).__name__
    if name in {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "ServiceUnavailableError",
        "ServerError",
    }:
        return True

    # Anything with an HTTP status attribute we can inspect.
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if isinstance(status, int) and (status == 429 or 500 <= status < 600):
        return True

    return False


async def _with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    provider: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_backoff: float = DEFAULT_INITIAL_BACKOFF,
    max_backoff: float = DEFAULT_MAX_BACKOFF,
) -> T:
    """Run an async operation with exponential backoff on transient errors.

    Args:
        operation: Zero-arg coroutine factory to invoke.
        provider: Provider name for log context.
        max_retries: Maximum retry attempts after the initial call.
        initial_backoff: Seconds to wait before the first retry.
        max_backoff: Cap on backoff between attempts.

    Returns:
        The operation's return value.

    Raises:
        The last exception if all attempts fail, or the original exception
        immediately if it is classified as non-transient.
    """
    attempt = 0
    while True:
        try:
            return await operation()
        except Exception as exc:
            if not _is_transient(exc) or attempt >= max_retries:
                raise
            backoff = min(max_backoff, initial_backoff * (2 ** attempt))
            # Jitter to avoid thundering herd on shared rate limits.
            backoff *= 0.5 + random.random()
            logger.warning(
                f"{provider} transient error (attempt {attempt + 1}/{max_retries + 1}): "
                f"{type(exc).__name__}: {exc}. Retrying in {backoff:.1f}s"
            )
            await asyncio.sleep(backoff)
            attempt += 1


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

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: Optional[str] = None):
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

    async def _do_complete(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
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

        # Use streaming to avoid timeout errors with the Anthropic SDK
        content = ""
        input_tokens = 0
        output_tokens = 0
        finish_reason = ""

        async with client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                content += text
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

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate completion using Claude with streaming."""
        return await _with_retry(
            lambda: self._do_complete(prompt, system_prompt, max_tokens, temperature),
            provider=self.provider_name,
        )


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

    async def _do_complete(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        client = self._get_client()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as first_err:
            # Some models (o-series, mini reasoning) reject temperature.
            # Retry once without it, then re-raise.
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

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate completion using OpenAI."""
        return await _with_retry(
            lambda: self._do_complete(prompt, system_prompt, max_tokens, temperature),
            provider=self.provider_name,
        )


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

    async def _do_complete(
        self,
        prompt: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> LLMResponse:
        from google.genai import types

        client = self._get_client()
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        if system_prompt:
            config.system_instruction = system_prompt

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

    async def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> LLMResponse:
        """Generate completion using Gemini."""
        return await _with_retry(
            lambda: self._do_complete(prompt, system_prompt, max_tokens, temperature),
            provider=self.provider_name,
        )


def get_llm_client(
    model: str = "claude-sonnet-4-6",
    api_key: Optional[str] = None,
) -> LLMClient:
    """
    Get an LLM client for the specified model.

    Automatically selects the correct provider based on the model name.

    Args:
        model: Model identifier (e.g., "claude-sonnet-4-6", "gpt-5.2", "gemini-2.5-flash")
        api_key: Optional API key (uses environment variable if not provided)

    Returns:
        LLMClient instance for the specified model

    Raises:
        ValueError: If model is not recognized

    Example:
        >>> client = get_llm_client("claude-sonnet-4-6")
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

    api_key = get_api_key(provider)
    if not api_key:
        key_names = {
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.GOOGLE: "GOOGLE_API_KEY",
        }
        return False, f"Missing API key: {key_names[provider]}"

    try:
        if provider == LLMProvider.ANTHROPIC:
            import anthropic  # noqa: F401
        elif provider == LLMProvider.OPENAI:
            import openai  # noqa: F401
        elif provider == LLMProvider.GOOGLE:
            from google import genai  # noqa: F401
    except ImportError as e:
        return False, f"Missing package: {e}"

    return True, ""
