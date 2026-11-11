import subprocess
import sys
import textwrap


def test_gammalake_works_without_ray_or_deltalake(tmp_path):
    script = textwrap.dedent(
        """
        import sys
        from datetime import UTC, datetime

        sys.modules["ray"] = sys.modules["deltalake"] = None

        import polars as pl

        from gammalake import GammaFeatureLake

        lake = GammaFeatureLake(base_path=sys.argv[1]).initialize()
        frame = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)],
                "symbol": ["AAPL", "AAPL"],
                "feature": [1.0, 2.0],
            }
        )
        lake.add_features(frame)
        assert lake.read(["feature"])["feature"].to_list() == [1.0, 2.0]
        assert lake.feature_metadata_frame().collect()["feature_name"].to_list() == ["feature"]
        """
    )

    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
