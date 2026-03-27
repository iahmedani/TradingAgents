import os
from typing import Any, Optional

from langchain_anthropic import ChatAnthropic

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model

_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "api_key", "max_tokens",
    "callbacks", "http_client", "http_async_client", "effort",
)


class NormalizedChatAnthropic(ChatAnthropic):
    """ChatAnthropic with normalized content output.

    Claude models with extended thinking or tool use return content as a
    list of typed blocks. This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        return normalize_content(super().invoke(input, config, **kwargs))


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic Claude models.

    Supports two authentication modes:
    1. API key: Set ANTHROPIC_API_KEY (direct API access)
    2. Account subscription: Set ANTHROPIC_BASE_URL and ANTHROPIC_AUTH_TOKEN
       to use a custom endpoint with bearer token authentication
    """

    def __init__(self, model: str, base_url: Optional[str] = None, **kwargs):
        super().__init__(model, base_url, **kwargs)

    def get_llm(self) -> Any:
        """Return configured ChatAnthropic instance."""
        llm_kwargs = {"model": self.model}

        # Resolve base URL: explicit param > env var > default Anthropic API
        base_url = self.base_url or os.environ.get("ANTHROPIC_BASE_URL")
        if base_url:
            llm_kwargs["anthropic_api_url"] = base_url

        for key in _PASSTHROUGH_KWARGS:
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # When using a custom base URL (account subscription / proxy),
        # the endpoint handles its own auth. If no ANTHROPIC_API_KEY is set,
        # provide a placeholder to satisfy the SDK's validation check.
        if base_url and "api_key" not in llm_kwargs and not os.environ.get("ANTHROPIC_API_KEY"):
            llm_kwargs["api_key"] = "not-needed"

        # If ANTHROPIC_AUTH_TOKEN is set, pass it as a Bearer token via
        # default_headers for endpoints that require token authentication.
        auth_token = os.environ.get("ANTHROPIC_AUTH_TOKEN")
        if auth_token:
            llm_kwargs.setdefault("default_headers", {})
            llm_kwargs["default_headers"]["Authorization"] = f"Bearer {auth_token}"

        return NormalizedChatAnthropic(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for Anthropic."""
        return validate_model("anthropic", self.model)
