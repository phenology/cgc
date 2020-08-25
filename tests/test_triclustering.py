import pytest

import numpy as np
import dask.array as da

from cgc.triclustering import Triclustering


@pytest.fixture
def triclustering():
    np.random.seed(1234)
    da.random.seed(1234)
    d, m, n = 3, 4, 3
    ncl_row, ncl_col, ncl_bnd = 2, 2, 2
    Z = np.random.randint(100, size=(d, m, n)).astype('float64')
    return Triclustering(Z, nclusters_row=ncl_row, nclusters_col=ncl_col,
                         nclusters_bnd=ncl_bnd, conv_threshold=1.e-5,
                         max_iterations=100, nruns=10, epsilon=1.e-8)


class TestTriclustering:
    def test_run_with_threads(self, triclustering):
        triclustering.run_with_threads(nthreads=2)
        assert isinstance(triclustering.row_clusters, np.ndarray)
        assert isinstance(triclustering.col_clusters, np.ndarray)
        assert isinstance(triclustering.bnd_clusters, np.ndarray)
        assert np.isclose(triclustering.error, -4638.9183068944785)

    def test_nruns_completed(self, triclustering):
        triclustering.run_with_threads(nthreads=1)
        assert triclustering.nruns_completed == 10
        triclustering.run_with_threads(nthreads=1)
        assert triclustering.nruns_completed == 20
