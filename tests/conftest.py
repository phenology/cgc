import pytest

from dask.distributed import Client, LocalCluster


@pytest.fixture(scope="session")
def dask_client():
    cluster = LocalCluster(n_workers=1, threads_per_worker=2)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()
