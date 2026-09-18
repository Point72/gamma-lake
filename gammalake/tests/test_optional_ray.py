import subprocess
import sys
import textwrap


def test_local_mode_works_without_ray(tmp_path):
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        from datetime import UTC, datetime

        class BlockRay(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                if fullname == "ray" or fullname.startswith("ray."):
                    raise ModuleNotFoundError("Ray import blocked by test", name="ray")
                return None

        sys.meta_path.insert(0, BlockRay())

        import polars as pl

        from gammalake import GammaFeatureLake

        lake = GammaFeatureLake(base_path=sys.argv[1]).initialize()
        frame = pl.DataFrame(
            {
                "timestamp": [datetime(2024, 1, 1, tzinfo=UTC)],
                "symbol": ["AAPL"],
                "feature": [1.0],
            }
        )
        lake.add_features(frame)
        assert lake.read(["feature"])["feature"].to_list() == [1.0]

        try:
            GammaFeatureLake(base_path=sys.argv[1], run_on_ray_cluster=True)
        except ImportError as exc:
            assert "gamma-lake[ray]" in str(exc)
        else:
            raise AssertionError("Ray mode should require the optional dependency")
        """
    )

    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, capture_output=True, text=True)
