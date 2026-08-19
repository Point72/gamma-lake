"""Lightweight tracing compatibility utilities."""

import functools
from collections.abc import Callable
from typing import Any

__all__ = ("trace",)


class trace:
    """No-op trace decorator."""

    def __init__(self, **kwargs: Any) -> None:
        pass

    def __call__(self, func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
