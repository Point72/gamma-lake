import pytest
import ray


class RayFixture:
    def __init__(self, init_kwargs):
        self._init_kwargs = init_kwargs

    def __enter__(self):
        if ray.is_initialized():
            ray.shutdown()
        ray.init(**self._init_kwargs)
        return self

    def __exit__(self, *args):
        ray.shutdown()


@pytest.fixture(scope="module")
def barebones_ray_cluster():
    with RayFixture({"address": "local", "num_cpus": 5}):
        yield None
