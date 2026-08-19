from typing import Annotated, Any, Protocol, TypeVar

import ray
from pydantic import InstanceOf
from pydantic.functional_validators import BeforeValidator
from typing_extensions import TypeAliasType

__all__ = ("Comparable", "ComparableType", "RayObjectReference")


class Comparable(Protocol):
    """A class which allows for comparisons with itself"""

    def __lt__(self, other) -> bool: ...
    def __le__(self, other) -> bool: ...
    def __gt__(self, other) -> bool: ...
    def __ge__(self, other) -> bool: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...


def _check_comparable(v):
    if v is not None and not hasattr(v, "__lt__"):
        raise ValueError(f"Value of type {type(v).__name__} is not comparable (missing __lt__)")
    return v


ComparableType = Annotated[Any, BeforeValidator(_check_comparable)]

T = TypeVar("T")
RayObjectReference = TypeAliasType("RayObjectReference", InstanceOf[ray.ObjectRef], type_params=(T,))
