"""GammaFeatureLake read and write benchmarks for airspeed velocity."""

import os
import re
import shutil
import uuid
import warnings
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import polars as pl

from gammalake import GammaFeatureLake
from gammalake.tests.base import generate_test_data

N_SYMBOLS = 200
N_FEATURES = 50
N_DAYS = 30

_ORIGIN = datetime(2024, 1, 1, tzinfo=UTC)
_SYMBOLS = [f"SYM_{i:04d}" for i in range(N_SYMBOLS)]
_ALL_DAYS = [_ORIGIN + timedelta(days=day) for day in range(N_DAYS)]
_GENERATED_DIR = "gammalake-asv"
_SCOPES = {"overlap-write", "read", "write"}
_RUN_NAME = re.compile(r"^\d{8}T\d{6}_[0-9a-f]{8}$")


def _local_root() -> Path:
    configured = os.environ.get("GAMMALAKE_BENCH_LOCAL_ROOT", ".asv/benchmark-data")
    return Path(configured).expanduser().resolve()


def _s3_base() -> str | None:
    configured = os.environ.get("GAMMALAKE_BENCH_S3_PREFIX")
    if configured is None:
        return None

    parsed = urlparse(configured)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/") or parsed.params or parsed.query or parsed.fragment:
        raise ValueError("GAMMALAKE_BENCH_S3_PREFIX must use the form s3://bucket/prefix")
    return f"s3://{parsed.netloc}/{parsed.path.strip('/')}"


def _make_feature_df(group_idx: int, n_features: int) -> pl.DataFrame:
    """Build a benchmark frame with deterministic feature values."""
    n_rows = N_DAYS * N_SYMBOLS
    rng = np.random.default_rng(group_idx)
    data = {
        "timestamp": [day for day in _ALL_DAYS for _ in range(N_SYMBOLS)],
        "symbol": _SYMBOLS * N_DAYS,
        **{f"g{group_idx}_feat_{feature}": rng.standard_normal(n_rows) for feature in range(n_features)},
    }
    return pl.DataFrame(data).with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))


def _new_run_path(scope: str) -> str:
    """Create a unique generated path for a benchmark run."""
    if scope not in _SCOPES:
        raise ValueError(f"Unsupported benchmark scope: {scope}")

    run_name = f"{datetime.now(UTC):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
    s3_base = _s3_base()
    if s3_base is not None:
        return f"{s3_base}/{_GENERATED_DIR}/{scope}/{run_name}"

    path = _local_root() / _GENERATED_DIR / scope / run_name
    path.mkdir(parents=True)
    return str(path)


def _is_generated_run(scope: str, run_name: str) -> bool:
    return scope in _SCOPES and _RUN_NAME.fullmatch(run_name) is not None


def _delete_local_path(storage_path: str) -> None:
    path = Path(storage_path).resolve()
    generated_root = (_local_root() / _GENERATED_DIR).resolve()
    if path.parent.parent != generated_root or not _is_generated_run(path.parent.name, path.name):
        raise ValueError(f"Refusing to remove non-benchmark path: {storage_path}")
    if path.exists():
        shutil.rmtree(path)


def _delete_s3_prefix(storage_path: str) -> None:
    import boto3

    s3_base = _s3_base()
    if s3_base is None or not storage_path.startswith(f"{s3_base}/{_GENERATED_DIR}/"):
        raise ValueError(f"Refusing to remove non-benchmark S3 prefix: {storage_path}")

    parsed = urlparse(storage_path)
    relative = parsed.path.strip("/")[len(urlparse(s3_base).path.strip("/")) :].strip("/")
    parts = relative.split("/")
    if len(parts) != 3 or parts[0] != _GENERATED_DIR or not _is_generated_run(parts[1], parts[2]):
        raise ValueError(f"Refusing to remove non-benchmark S3 prefix: {storage_path}")

    client = boto3.client("s3")
    prefix = f"{parsed.path.strip('/')}/"
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=parsed.netloc, Prefix=prefix):
        objects = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=parsed.netloc, Delete={"Objects": objects})


def _delete_generated_path(storage_path: str) -> None:
    """Remove only paths created by this benchmark module."""
    try:
        if storage_path.startswith("s3://"):
            _delete_s3_prefix(storage_path)
        else:
            _delete_local_path(storage_path)
    except (ImportError, OSError, ValueError) as exc:
        warnings.warn(f"Unable to clean up benchmark data at {storage_path}: {exc}", stacklevel=2)


class GammaLakeWriteSuite:
    """Benchmark sequential ``GammaFeatureLake.add_features`` merge scenarios."""

    number = 1
    repeat = 1
    rounds = 1
    warmup_time = 0

    N_FEATURES = 3
    N_SYMBOLS = 5
    N_DAYS = 10
    START_DATE = datetime(2020, 6, 20, tzinfo=UTC)

    @classmethod
    def setup_cache(cls) -> dict[str, str]:
        storage_path = _new_run_path("write")
        try:
            GammaFeatureLake(base_path=storage_path, run_on_ray_cluster=False).initialize()
        except Exception:
            _delete_generated_path(storage_path)
            raise
        return {"storage_path": storage_path}

    def setup(self, cache: dict[str, str]) -> None:
        self._lake = GammaFeatureLake(base_path=cache["storage_path"], run_on_ray_cluster=False)

    def time_step_01_initial_write(self, _cache: dict[str, str]) -> None:
        frame = generate_test_data(
            n_features=self.N_FEATURES,
            n_symbols=self.N_SYMBOLS,
            n_days=self.N_DAYS,
            start_date=self.START_DATE,
        )
        self._lake.add_features(frame, owner="benchmark", overlap_mode="merge")

    def time_step_02_expand_symbols(self, _cache: dict[str, str]) -> None:
        frame = generate_test_data(
            n_features=self.N_FEATURES,
            n_symbols=2 * self.N_SYMBOLS,
            n_days=self.N_DAYS,
            start_date=self.START_DATE,
        )
        self._lake.add_features(frame, owner="benchmark", overlap_mode="merge")

    def time_step_03_new_symbols(self, _cache: dict[str, str]) -> None:
        frame = generate_test_data(
            n_features=self.N_FEATURES,
            n_symbols=10,
            symbols_id_start=10,
            n_days=self.N_DAYS,
            start_date=self.START_DATE,
        )
        self._lake.add_features(frame, owner="benchmark", overlap_mode="merge")

    def time_step_04_future_symbols(self, _cache: dict[str, str]) -> None:
        frame = generate_test_data(
            n_features=self.N_FEATURES,
            n_symbols=20,
            symbols_id_start=20,
            n_days=self.N_DAYS,
            start_date=self.START_DATE + timedelta(days=20),
        )
        self._lake.add_features(frame, owner="benchmark", overlap_mode="merge")

    def time_step_05_future_features(self, _cache: dict[str, str]) -> None:
        frame = generate_test_data(
            n_features=self.N_FEATURES,
            n_symbols=20,
            symbols_id_start=40,
            feature_ids_start=self.N_FEATURES,
            n_days=self.N_DAYS,
            start_date=self.START_DATE + timedelta(days=20),
        )
        self._lake.add_features(frame, owner="benchmark", overlap_mode="merge")

    def time_step_06_new_symbols_and_features(self, _cache: dict[str, str]) -> None:
        frame = generate_test_data(
            n_features=self.N_FEATURES,
            n_symbols=2 * self.N_SYMBOLS,
            symbols_id_start=55,
            feature_ids_start=2 * self.N_FEATURES,
            n_days=self.N_DAYS,
            start_date=self.START_DATE + timedelta(days=20),
        )
        self._lake.add_features(frame, owner="benchmark", overlap_mode="merge")

    def track_zz_cleanup(self, cache: dict[str, str]) -> int:
        _delete_generated_path(cache["storage_path"])
        return 0


class GammaLakeOverlapWriteSuite:
    """Benchmark ``add_features`` with heavily overlapping time intervals."""

    number = 1
    repeat = 1
    rounds = 1
    timeout = 600
    warmup_time = 0

    START_DATE = datetime(2020, 6, 20, tzinfo=UTC)

    @classmethod
    def setup_cache(cls) -> dict[str, str]:
        storage_path = _new_run_path("overlap-write")
        try:
            GammaFeatureLake(base_path=storage_path, run_on_ray_cluster=False).initialize()
        except Exception:
            _delete_generated_path(storage_path)
            raise
        return {"storage_path": storage_path}

    def setup(self, cache: dict[str, str]) -> None:
        self._lake = GammaFeatureLake(base_path=cache["storage_path"], run_on_ray_cluster=False)

    def _add_features(self, **kwargs) -> None:
        self._lake.add_features(generate_test_data(**kwargs), owner="benchmark", overlap_mode="merge")

    def time_step_01_initial_write(self, _cache: dict[str, str]) -> None:
        self._add_features(n_features=N_FEATURES, n_symbols=N_SYMBOLS, n_days=N_DAYS, start_date=self.START_DATE)

    def time_step_02_full_overlap(self, _cache: dict[str, str]) -> None:
        self._add_features(n_features=N_FEATURES, n_symbols=N_SYMBOLS, n_days=N_DAYS, start_date=self.START_DATE)

    def time_step_03_new_features(self, _cache: dict[str, str]) -> None:
        self._add_features(
            n_features=N_FEATURES,
            n_symbols=N_SYMBOLS,
            n_days=N_DAYS,
            start_date=self.START_DATE,
            feature_ids_start=N_FEATURES,
        )

    def time_step_04_forward_write(self, _cache: dict[str, str]) -> None:
        self._add_features(
            n_features=N_FEATURES,
            n_symbols=N_SYMBOLS,
            n_days=N_DAYS,
            start_date=self.START_DATE + timedelta(days=N_DAYS),
        )

    def time_step_05_partial_overlap(self, _cache: dict[str, str]) -> None:
        self._add_features(
            n_features=N_FEATURES,
            n_symbols=N_SYMBOLS,
            n_days=N_DAYS,
            start_date=self.START_DATE + timedelta(days=N_DAYS // 2),
        )

    def time_step_06_partial_new_features(self, _cache: dict[str, str]) -> None:
        self._add_features(
            n_features=N_FEATURES,
            n_symbols=N_SYMBOLS,
            n_days=N_DAYS,
            start_date=self.START_DATE + timedelta(days=N_DAYS // 2),
            feature_ids_start=N_FEATURES,
        )

    def time_step_07_full_remerge(self, _cache: dict[str, str]) -> None:
        self._add_features(n_features=N_FEATURES, n_symbols=N_SYMBOLS, n_days=2 * N_DAYS, start_date=self.START_DATE)

    def time_step_08_new_symbols(self, _cache: dict[str, str]) -> None:
        self._add_features(
            n_features=N_FEATURES,
            n_symbols=N_SYMBOLS,
            n_days=2 * N_DAYS,
            start_date=self.START_DATE,
            symbols_id_start=N_SYMBOLS,
        )

    def track_zz_cleanup(self, cache: dict[str, str]) -> int:
        _delete_generated_path(cache["storage_path"])
        return 0


class GammaLakeReadSuite:
    """Benchmark reads over multiple time windows and feature-group counts."""

    MAX_GROUPS = 3
    params = ([1, 7, 30], [1, 3])
    param_names = ("n_days", "n_groups")
    timeout = 600

    def setup(self, n_days: int, n_groups: int) -> None:
        self._storage_path = _new_run_path("read")
        try:
            self._lake = GammaFeatureLake(base_path=self._storage_path, run_on_ray_cluster=False).initialize()
            features: list[str] = []
            for group in range(self.MAX_GROUPS):
                self._lake.add_features(_make_feature_df(group, N_FEATURES), owner="benchmark")
                features.extend(f"g{group}_feat_{feature}" for feature in range(N_FEATURES))
        except Exception:
            _delete_generated_path(self._storage_path)
            raise
        self._features = features[: n_groups * N_FEATURES]
        self._start = _ALL_DAYS[0]
        self._end = _ALL_DAYS[n_days - 1]

    def time_read(self, n_days: int, n_groups: int) -> None:
        self._lake.read(self._features, start=self._start, end=self._end)

    def teardown(self, n_days: int, n_groups: int) -> None:
        _delete_generated_path(self._storage_path)
