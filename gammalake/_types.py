from typing import Protocol, TypeVar

import ray
from pydantic import InstanceOf
from typing_extensions import TypeAliasType

__all__ = ("Comparable", "RayObjectReference")


class Comparable(Protocol):
    """A class which allows for comparisons with itself"""

    def __lt__(self, other) -> bool: ...
    def __le__(self, other) -> bool: ...
    def __gt__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...


T = TypeVar("T")
RayObjectReference = TypeAliasType("RayObjectReference", InstanceOf[ray.ObjectRef], type_params=(T,))
