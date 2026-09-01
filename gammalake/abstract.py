import abc
from enum import Enum
from functools import singledispatchmethod
from typing import Literal

import polars as pl
import ray
from ccflow import BaseModel
from pydantic import Field, model_validator

from gammalake._types import Comparable

__all__ = (
    "BaseFeatureLake",
    "Comparable",
    "DefaultUploadMergeMode",
    "InPlaceFeatureModifier",
    "MissingFeaturesException",
    "MissingOrMisregisteredSignalsException",
    "NoOpModifier",
    "UninitializedDeltaLakeException",
    "UploadMergeModes",
)


class MissingFeaturesException(KeyError):
    """Exception raised when a user tries to read features which do not exist in the specified GammaFeatureLake or provided meta table."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class MismatchedIndexSchema(ValueError):
    """Exception raised when a user's provided IndexSchema does not match a preexisting Index's Schema"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class UninitializedDeltaLakeException(ValueError):
    """Exception raised when a user attempts to use a feature store without ever initializing it"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class MissingOrMisregisteredSignalsException(ValueError):
    """Exception raised when a user attempts to read targets which are either missing or registered as Features"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


UploadMergeModes = Literal["append", "upsert", "overwrite", "increment"]
DefaultUploadMergeMode = "increment"


class StatusCode(Enum):
    SUCCESS = 1
    FAILURE = 2
    OTHER = 3


class BaseFeatureLake(abc.ABC):
    """
    Abstract base class for feature lake operations.

    This class defines the interface for interacting with feature lakes,
    providing abstract methods for querying and uploading features. Subclasses
    must implement these methods to handle specific data operations.
    """

    @abc.abstractmethod
    def table_metadata_frame(self, start: Comparable | None = None, end: Comparable | None = None) -> pl.LazyFrame:
        """Returns the table-level meta table. Used in query operations."""

    @abc.abstractmethod
    def feature_metadata_frame(self, start: Comparable | None = None, end: Comparable | None = None) -> pl.LazyFrame:
        """Returns the feature-level meta table. Used in query operations."""

    @abc.abstractmethod
    def index_frame(self, start: Comparable | None = None, end: Comparable | None = None) -> pl.LazyFrame:
        """Returns the index table. Used in query operations."""

    @singledispatchmethod
    def read(self, *arg) -> pl.DataFrame:
        """A single dispatch method used to read features from this FeatureLake. See the specializations for more details."""

    @read.register
    @abc.abstractmethod
    def _(self, features: list, start: Comparable | None = None, end: Comparable | None = None, local: bool = True) -> pl.DataFrame:
        """A specialization of .read(...) which takes a list of feature names as input, a start/end value for comparisons, and a boolean value 'local' indicating
        whether or not the user wants to take advantage of parallelization on a ray cluster for this operation.

        .. code-block:: python

            fs = FeatureLake(...)
            features = fs.read(["feature1", "feature2", start=datetime.date(2020,1,1), end=datetime.date(2020,2,1))

        Args:
            features: a list of feature names found in this FeatureLake's feature metadata table.
            start: A comparable value, the feature values of which will have their index entries >= start.
            end: A comparable value, the feature values of which will have their index entries <= end.
            local: A flag indicating whether or not to use ray to parallelize the read operations.

        Returns: a polars DataFrame of indexed features
        """

    @read.register
    @abc.abstractmethod
    def _(
        self, features: pl.DataFrame | pl.LazyFrame, start: Comparable | None = None, end: Comparable | None = None, local: bool = True
    ) -> pl.DataFrame:
        """A specialization of .read(...) which takes a polars DataFrame (or LazyFrame) as input, a start/end value for comparisons, and a boolean value 'local' indicating
        whether or not the user wants to take advantage of parallelization on a ray cluster for this operation. The input Data/LazyFrame is expected to be a filtered
        version of this class' feature_metadata_frame(...) method, and will read features based the rows present in that frame.

        .. code-block:: python

            fs = FeatureLake(...)
            features = fs.read(fs.feature_metadata_frame().filter(pl.col("owner") = "my_user_name")), start=datetime.date(2020,1,1), end=datetime.date(2020,2,1))

        Args:
            features: a possibly filtered version of this object's feature metadata table, used to identify which features to read and return to the user.
            start: A comparable value, the feature values of which will have their index entries >= start.
            end: A comparable value, the feature values of which will have their index entries <= end.
            local: A flag indicating whether or not to use ray to parallelize the read operations.

        Returns: a polars DataFrame of indexed features
        """

    @abc.abstractmethod
    def add_features(
        self,
        df: pl.DataFrame | ray.ObjectRef,
        owner: str = "missing_owner",
        metadata: pl.DataFrame | None = None,
    ) -> list[StatusCode]:
        """Add features to this FeatureLake object. An optional metadata frame argument allows users to increment very specific feature versions, otherwise we fallback
        to generic insertion logic around latest feature version value.

        Args:
            df: an input polars DataFrame (or object reference pointing to a polars DataFrame) containing feature values to be added to this FeatureLake's underlying DeltaTables.
            owner: a string argument for use in identifying ownership in metadata tables.
            metadata: an optional DataFrame object, expected to be a (possibly) filtered version of this object's feature metadata frame, identifying which DeltaTables to update

        Returns: A List of StatusCode enums
        """
        ...

    @abc.abstractmethod
    def add_targets(
        self,
        df: pl.DataFrame | ray.ObjectRef,
        owner: str = "missing_owner",
        metadata: pl.DataFrame | None = None,
    ) -> list[StatusCode]:
        """Add targets to this FeatureLake object. An optional metadata frame argument allows users to increment very specific feature versions, otherwise we fallback
        to generic insertion logic around latest feature version value.

        Args:
            df: an input polars DataFrame (or object reference pointing to a polars DataFrame) containing feature values to be added to this FeatureLake's underlying DeltaTables.
            owner: a string argument for use in identifying ownership in metadata tables.
            metadata: an optional DataFrame object, expected to be a (possibly) filtered version of this object's feature metadata frame, identifying which DeltaTables to update

        Returns: A List of StatusCode enums
        """
        ...


class FeatureMetadata(BaseModel):
    """A wrapper class with convenience methods for interfacing with onnx serialized models and a GammaFeatureLake"""

    feature_names: list[str] = Field
    feature_versions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def check(self) -> "FeatureMetadata":
        """
        Verify alignment between the number of feature_names and versions - either match exactly, or feature_versions should be an empty list.
        """
        if len(self.feature_names) != len(self.feature_versions):
            assert self.feature_versions == [], "Provided feature versions should match in length with the feature names, or be empty"
        return self

    def get_metadata_frame(self, fs: BaseFeatureLake):
        """filter a feature metadata table based on the names/optional version entries in this object"""
        if len(self.feature_versions) == 0:
            df = fs._get_latest_feature_tables(self.feature_names, fs.feature_metadata_frame().collect())
        else:
            filter_df = pl.DataFrame(
                [(name, version) for name, version in zip(self.feature_names, self.feature_versions)],
                schema=["feature_name", "version"],
            )
            df = fs.feature_metadata_frame().collect().join(filter_df, how="inner", on=["feature_name", "version"])

        if df.height != len(self.feature_names):
            raise ValueError("Provided feature/version information was not found in the provided FeatureLake!")
        return df


class InPlaceFeatureModifier(BaseModel):
    """
    A callable class which modifies a dataframe in-place.
    """

    @abc.abstractmethod
    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """Modify a polars dataframe in-place"""
        ...


class NoOpModifier(InPlaceFeatureModifier):
    """
    A callable class which performs a trivial 'modification'
    """

    def __call__(self, df: pl.DataFrame) -> pl.DataFrame:
        """Return the input dataframe"""
        return df
