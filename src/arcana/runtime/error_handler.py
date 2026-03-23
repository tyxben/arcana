"""Error handling strategies for runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from arcana.runtime.exceptions import ErrorSeverity, ErrorType, RuntimeError

if TYPE_CHECKING:
    from arcana.contracts.state import AgentState


class RetryStrategy:
    """Configuration for retry behavior."""

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        initial_delay_ms: int = 1000,
        max_delay_ms: int = 10000,
        backoff_multiplier: float = 2.0,
        jitter: bool = True,
    ) -> None:
        """
        Initialize retry strategy.

        Args:
            max_attempts: Maximum retry attempts
            initial_delay_ms: Initial delay in milliseconds
            max_delay_ms: Maximum delay in milliseconds
            backoff_multiplier: Exponential backoff multiplier
            jitter: Add random jitter to delays
        """
        self.max_attempts = max_attempts
        self.initial_delay_ms = initial_delay_ms
        self.max_delay_ms = max_delay_ms
        self.backoff_multiplier = backoff_multiplier
        self.jitter = jitter

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt.

        Args:
            attempt: Attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        import random

        delay_ms = min(
            self.initial_delay_ms * (self.backoff_multiplier**attempt),
            self.max_delay_ms,
        )

        if self.jitter:
            # Add random jitter ±25%
            jitter_range = delay_ms * 0.25
            delay_ms += random.uniform(-jitter_range, jitter_range)

        return delay_ms / 1000


class ErrorHandler:
    """
    Handles errors during agent execution.

    Implements:
    - Retry logic for recoverable errors
    - Escalation for unrecoverable errors
    - Error classification and logging
    """

    def __init__(
        self,
        *,
        retry_strategy: RetryStrategy | None = None,
        escalation_callback: Callable[[RuntimeError, AgentState], None] | None = None,
    ) -> None:
        """
        Initialize error handler.

        Args:
            retry_strategy: Strategy for retries (default if None)
            escalation_callback: Callback for escalated errors
        """
        self.retry_strategy = retry_strategy or RetryStrategy()
        self.escalation_callback = escalation_callback

    async def handle_error(
        self,
        error: Exception,
        state: AgentState,
        *,
        context: str | None = None,
    ) -> tuple[bool, str | None]:
        """
        Handle an error and decide action.

        Args:
            error: The error that occurred
            state: Current agent state
            context: Optional context about where error occurred

        Returns:
            (should_retry, error_message)
        """
        # Convert to RuntimeError if not already
        if not isinstance(error, RuntimeError):
            runtime_error = self._classify_error(error)
        else:
            runtime_error = error

        # Decide action based on error type
        if runtime_error.error_type == ErrorType.RETRYABLE:
            return (True, None)

        elif runtime_error.error_type == ErrorType.VALIDATION:
            # Validation errors can be retried with correction
            return (True, f"Validation failed: {runtime_error.message}")

        elif runtime_error.error_type == ErrorType.PARTIAL_FAILURE:
            # Continue but log the issue
            return (False, f"Partial failure: {runtime_error.message}")

        elif runtime_error.error_type in {
            ErrorType.BUDGET_EXCEEDED,
            ErrorType.PERMANENT,
        }:
            # Cannot retry
            return (False, runtime_error.message)

        elif runtime_error.error_type in {
            ErrorType.REQUIRES_HUMAN,
            ErrorType.SAFETY_VIOLATION,
            ErrorType.AUTHORIZATION,
        }:
            # Escalate
            if self.escalation_callback:
                self.escalation_callback(runtime_error, state)
            return (False, f"Escalation required: {runtime_error.message}")

        # Default: don't retry
        return (False, runtime_error.message)

    def _classify_error(self, error: Exception) -> RuntimeError:
        """
        Classify a generic exception into a RuntimeError.

        Args:
            error: Exception to classify

        Returns:
            RuntimeError with appropriate classification
        """
        error_str = str(error).lower()

        # Rate limits and timeouts
        if any(x in error_str for x in ["rate limit", "429", "too many requests"]):
            return RuntimeError(
                str(error),
                recoverable=True,
                error_type=ErrorType.RETRYABLE,
                severity=ErrorSeverity.MEDIUM,
            )

        if any(x in error_str for x in ["timeout", "timed out", "503"]):
            return RuntimeError(
                str(error),
                recoverable=True,
                error_type=ErrorType.RETRYABLE,
                severity=ErrorSeverity.MEDIUM,
            )

        # Budget exceeded
        if any(x in error_str for x in ["budget", "quota", "limit exceeded"]):
            return RuntimeError(
                str(error),
                recoverable=False,
                error_type=ErrorType.BUDGET_EXCEEDED,
                severity=ErrorSeverity.HIGH,
            )

        # Authorization
        if any(x in error_str for x in ["unauthorized", "forbidden", "401", "403"]):
            return RuntimeError(
                str(error),
                recoverable=False,
                error_type=ErrorType.AUTHORIZATION,
                severity=ErrorSeverity.HIGH,
            )

        # Validation
        if any(x in error_str for x in ["validation", "invalid", "malformed"]):
            return RuntimeError(
                str(error),
                recoverable=True,
                error_type=ErrorType.VALIDATION,
                severity=ErrorSeverity.LOW,
            )

        # Default: permanent error
        return RuntimeError(
            str(error),
            recoverable=False,
            error_type=ErrorType.PERMANENT,
            severity=ErrorSeverity.MEDIUM,
        )

    async def execute_with_retry(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[bool, Any, Exception | None]:
        """
        Execute a function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            (success, result, error)
        """
        last_error = None

        for attempt in range(self.retry_strategy.max_attempts):
            try:
                result = await func(*args, **kwargs)
                return (True, result, None)

            except Exception as e:
                last_error = e

                # Check if should retry
                runtime_error = (
                    e if isinstance(e, RuntimeError) else self._classify_error(e)
                )

                if not runtime_error.recoverable:
                    return (False, None, e)

                # Wait before retry (except on last attempt)
                if attempt < self.retry_strategy.max_attempts - 1:
                    delay = self.retry_strategy.get_delay(attempt)
                    await asyncio.sleep(delay)

        # All retries exhausted
        return (False, None, last_error)
