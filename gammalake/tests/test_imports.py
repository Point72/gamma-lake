import subprocess
import sys
import textwrap


def test_gammalake_works_without_ray_or_deltalake(tmp_path):
    script = textwrap.dedent(
        """
        import importlib.abc
        import sys
        from datetime import UTC, datetime

        BLOCKED = ("ray", "deltalake")

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                root = fullname.partition(".")[0]
                if root in BLOCKED:
                    raise ModuleNotFoundError(f"{root} import blocked by test", name=root)
                return None

        sys.meta_path.insert(0, Block())

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
        assert not [name for name in sys.modules if name.partition(".")[0] in BLOCKED]
        """
    )

    subprocess.run([sys.executable, "-c", script, str(tmp_path)], check=True, capture_output=True, text=True)
