"""llm-gateway: one interface across LLM providers, with failover and cost tracking.

Quickstart::

    from llm_gateway import Gateway, AnthropicProvider, OpenAIProvider

    gateway = Gateway(
        [AnthropicProvider("claude-sonnet-4-5"), OpenAIProvider("gpt-5.4")],
        budget_usd=5.00,
    )
    reply = gateway.complete("Explain RAG in one sentence.")
    print(reply.text, reply.provider, reply.usage.cost_usd)
"""

from .errors import (
    AllProvidersFailed,
    AuthenticationError,
    BudgetExceeded,
    GatewayError,
    InvalidRequest,
    ProviderError,
    ProviderTimeout,
    ProviderUnavailable,
    RateLimited,
)
from .gateway import Gateway
from .ledger import Ledger, LedgerEntry
from .policy import CircuitBreaker, RetryPolicy
from .pricing import ModelPrice, PriceBook
from .providers import (
    AnthropicProvider,
    BaseProvider,
    GoogleProvider,
    OpenAIProvider,
    ParsedResponse,
    PreparedRequest,
)
from .types import Attempt, Completion, Message, Usage

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # core
    "Gateway",
    "Ledger",
    "LedgerEntry",
    "RetryPolicy",
    "CircuitBreaker",
    "PriceBook",
    "ModelPrice",
    # providers
    "BaseProvider",
    "AnthropicProvider",
    "OpenAIProvider",
    "GoogleProvider",
    "PreparedRequest",
    "ParsedResponse",
    # types
    "Message",
    "Completion",
    "Usage",
    "Attempt",
    # errors
    "GatewayError",
    "ProviderError",
    "RateLimited",
    "ProviderUnavailable",
    "ProviderTimeout",
    "AuthenticationError",
    "InvalidRequest",
    "AllProvidersFailed",
    "BudgetExceeded",
]
