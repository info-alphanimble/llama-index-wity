"""LlamaIndex integration for Wity, the typed-decision API."""

from wity import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    Choice,
    ChoiceAnswer,
    InsufficientBalanceError,
    InternalServerError,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Question,
    RateLimitError,
    Reasoning,
    ReasoningInfo,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
    WityError,
    choice,
    noul,
    score,
)

from .classifier import WityClassifier
from .selector import WitySelector
from .tool_gate import WityToolGate

__version__ = "0.1.0"

__all__ = [
    "APIConnectionError",
    "APIError",
    "APITimeoutError",
    "AuthenticationError",
    "BadRequestError",
    "Choice",
    "ChoiceAnswer",
    "InsufficientBalanceError",
    "InternalServerError",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "Question",
    "RateLimitError",
    "Reasoning",
    "ReasoningInfo",
    "Score",
    "ScoreAnswer",
    "SystemOneResponse",
    "Usage",
    "WityClassifier",
    "WityError",
    "WitySelector",
    "WityToolGate",
    "choice",
    "noul",
    "score",
]
