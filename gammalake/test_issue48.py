from datetime import UTC, datetime

import polars as pl
from polars.testing import assert_frame_equal

from gammalake import GammaFeatureLake


def test_write_at_last_updated_timestamp_keeps_feature_rows_aligned(tmp_path):
    def frame(rows, **features):
        return pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, day, tzinfo=UTC) for day, _ in rows],
                "symbol": [symbol for _, symbol in rows],
                **{name: [float(scale * day) for day, _ in rows] for name, scale in features.items()},
            }
        ).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))

    lake = GammaFeatureLake(base_path=str(tmp_path), run_on_ray_cluster=False).initialize()
    lake.add_features(frame([(1, "A"), (2, "A"), (3, "A")], x=1))
    lake.add_features(frame([(3, "B")], y=10))
    lake.add_features(frame([(4, "A")], x=1))

    result = lake.read(["x", "y"]).select("timestamp", "symbol", "x")
    expected = pl.DataFrame(
        {
            "timestamp": [datetime(2024, 1, day, tzinfo=UTC) for day in (1, 2, 3, 3, 4)],
            "symbol": ["A", "A", "A", "B", "A"],
            "x": [1.0, 2.0, 3.0, None, 4.0],
        }
    ).with_columns(pl.col("timestamp").dt.cast_time_unit("us"))
    assert_frame_equal(result, expected)
