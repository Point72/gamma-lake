from datetime import UTC, datetime

import polars as pl

from gammalake import GammaFeatureLake


def test_copy_overlap_preserves_values_after_overwritten_rows(tmp_path):
    lake = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()
    days = [datetime(2024, 1, day, tzinfo=UTC) for day in range(1, 4)]
    initial = pl.DataFrame(
        {
            "timestamp": [day for day in days for _ in ("A", "B")],
            "symbol": ["A", "B"] * len(days),
            "f1": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        }
    )
    lake.add_features(initial)
    lake.add_features(initial.filter((pl.col("timestamp") == days[1]) & (pl.col("symbol") == "A")).with_columns(f1=100.0))

    result = lake.read(["f1"]).sort(lake.sort_keys)

    assert result["f1"].to_list() == [1.0, 1.0, 100.0, 2.0, 3.0, 3.0], f"Expected preserved values after overwrite, got {result['f1'].to_list()}"
