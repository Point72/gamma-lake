from collections.abc import Callable
from importlib import import_module
from types import ModuleType
from typing import Any

__all__ = ("get", "is_object_ref", "remote", "require_ray", "wait")

_INSTALL_MESSAGE = "Ray support requires the optional dependency. Install it with `pip install 'gamma-lake[ray]'`."


class RayNotInstalledError(ImportError):
    """Raised when distributed execution is requested without Ray installed."""


def require_ray() -> ModuleType:
    """Import Ray or explain how to enable distributed execution."""
    try:
        return import_module("ray")
    except ModuleNotFoundError as exc:
        if exc.name != "ray":
            raise
        raise RayNotInstalledError(_INSTALL_MESSAGE) from exc


def is_object_ref(value: Any) -> bool:
    """Return whether a value is a Ray object reference without requiring Ray."""
    try:
        ray = require_ray()
    except RayNotInstalledError:
        return False
    return isinstance(value, ray.ObjectRef)


def remote(function: Callable, num_returns: int = 1, **kwargs) -> Callable:
    """Wrap a function as a Ray remote call."""
    return require_ray().remote(num_returns=num_returns, **kwargs)(function).remote


def get(value: Any) -> Any:
    """Resolve one or more Ray object references."""
    return require_ray().get(value)


def wait(values: list, **kwargs) -> tuple[list, list]:
    """Wait for Ray object references."""
    return require_ray().wait(values, **kwargs)
