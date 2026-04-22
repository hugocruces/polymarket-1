"""Tests for the LLM retry/backoff helper."""

import asyncio

import pytest

from polymarket_agent.llm_assessment import providers
from polymarket_agent.llm_assessment.providers import _is_transient, _with_retry


class TestIsTransient:
    def test_timeout_is_transient(self):
        assert _is_transient(asyncio.TimeoutError()) is True
        assert _is_transient(TimeoutError()) is True

    def test_connection_is_transient(self):
        assert _is_transient(ConnectionError("broken pipe")) is True

    def test_rate_limit_name_is_transient(self):
        class RateLimitError(Exception):
            pass

        assert _is_transient(RateLimitError()) is True

    def test_5xx_status_is_transient(self):
        class HTTPErr(Exception):
            def __init__(self):
                self.status_code = 503

        assert _is_transient(HTTPErr()) is True

    def test_429_status_is_transient(self):
        class HTTPErr(Exception):
            def __init__(self):
                self.status_code = 429

        assert _is_transient(HTTPErr()) is True

    def test_400_is_not_transient(self):
        class HTTPErr(Exception):
            def __init__(self):
                self.status_code = 400

        assert _is_transient(HTTPErr()) is False

    def test_value_error_is_not_transient(self):
        assert _is_transient(ValueError("nope")) is False


class TestWithRetry:
    @pytest.mark.asyncio
    async def test_returns_on_first_success(self):
        calls = 0

        async def op():
            nonlocal calls
            calls += 1
            return "ok"

        result = await _with_retry(op, provider="test", initial_backoff=0.0)
        assert result == "ok"
        assert calls == 1

    @pytest.mark.asyncio
    async def test_retries_transient_then_succeeds(self, monkeypatch):
        # Skip real sleep — we don't want tests to wait.
        async def no_sleep(_):
            return None

        monkeypatch.setattr(providers.asyncio, "sleep", no_sleep)

        calls = 0

        async def op():
            nonlocal calls
            calls += 1
            if calls < 3:
                raise TimeoutError("transient")
            return "ok"

        result = await _with_retry(
            op, provider="test", initial_backoff=0.0, max_retries=3
        )
        assert result == "ok"
        assert calls == 3

    @pytest.mark.asyncio
    async def test_reraises_non_transient_immediately(self):
        calls = 0

        async def op():
            nonlocal calls
            calls += 1
            raise ValueError("permanent")

        with pytest.raises(ValueError):
            await _with_retry(op, provider="test", initial_backoff=0.0)

        assert calls == 1

    @pytest.mark.asyncio
    async def test_gives_up_after_max_retries(self, monkeypatch):
        async def no_sleep(_):
            return None

        monkeypatch.setattr(providers.asyncio, "sleep", no_sleep)

        calls = 0

        async def op():
            nonlocal calls
            calls += 1
            raise TimeoutError("always")

        with pytest.raises(TimeoutError):
            await _with_retry(
                op, provider="test", initial_backoff=0.0, max_retries=2
            )

        # 1 initial attempt + 2 retries = 3
        assert calls == 3
