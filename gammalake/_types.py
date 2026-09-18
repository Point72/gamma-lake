from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from ray import ObjectRef as RayObjectReference
else:
    T = TypeVar("T")

    class RayObjectReference(Protocol, Generic[T]):
        """Static stand-in for Ray's generic ObjectRef when Ray is unavailable."""


__all__ = ("Comparable", "RayObjectReference")


class Comparable(Protocol):
    """A class which allows for comparisons with itself"""

    def __lt__(self, other) -> bool: ...
    def __le__(self, other) -> bool: ...
    def __gt__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
