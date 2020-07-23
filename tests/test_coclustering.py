import pytest

import numpy as np
import dask.array as da

from geoclustering.coclustering import Coclustering


@pytest.fixture
def coclustering():
    np.random.seed(1234)
    da.random.seed(1234)
    m, n = 4, 3
    ncl_row, ncl_col = 2, 2
    Z = np.random.randint(100, size=(m, n)).astype('float64')
    return Coclustering(Z, nclusters_row=ncl_row, nclusters_col=ncl_col,
                        conv_threshold=1.e-5, max_iterations=100, nruns=10,
                        epsilon=1.e-8)


class TestCoclustering:
    def test_run_with_threads(self, coclustering):
        coclustering.run_with_threads(nthreads=2)
        assert isinstance(coclustering.row_clusters, np.ndarray)
        assert isinstance(coclustering.col_clusters, np.ndarray)
        assert np.isclose(coclustering.error, -1430.4432279784644)

    def test_dask_runs_memory(self, client, coclustering):
        coclustering._dask_runs_memory()
        assert isinstance(coclustering.row_clusters, np.ndarray)
        assert isinstance(coclustering.col_clusters, np.ndarray)
        assert np.isclose(coclustering.error, -1430.4432279784644)

    def test_dask_runs_performance(self, client, coclustering):
        coclustering.client = client
        coclustering._dask_runs_performance()
        assert isinstance(coclustering.row_clusters, np.ndarray)
        assert isinstance(coclustering.col_clusters, np.ndarray)
        assert np.isclose(coclustering.error, -1430.4432279784644)
