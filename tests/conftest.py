import shutil
import pytest

from dask.distributed import Client, LocalCluster


@pytest.fixture(scope="session")
def dask_client():
    dask_local_path = 'dask-worker-space'
    cluster = LocalCluster(n_workers=1, threads_per_worker=2,
                           local_directory=dask_local_path)
    client = Client(cluster)
    yield client
    # teardown
    client.close()
    cluster.close()
    shutil.rmtree(dask_local_path)
