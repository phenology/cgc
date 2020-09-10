import pytest

import numpy as np
import dask.array as da

from cgc.triclustering import Triclustering


@pytest.fixture
def triclustering():
    np.random.seed(1234)
    da.random.seed(1234)
    d, m, n = 6, 10, 8
    ncl_row, ncl_col, ncl_bnd = 5, 2, 3
    Z = np.random.randint(100, size=(d, m, n)).astype('float64')
    return Triclustering(Z, nclusters_row=ncl_row, nclusters_col=ncl_col,
                         nclusters_bnd=ncl_bnd, conv_threshold=1.e-5,
                         max_iterations=100, nruns=10, epsilon=1.e-8)


class TestTriclustering:
    def test_check_initial_assignments(self, triclustering):
        assert triclustering.row_clusters is None
        assert triclustering.col_clusters is None
        assert triclustering.bnd_clusters is None

    def test_run_with_threads(self, triclustering):
        triclustering.set_initial_clusters([0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                                           [0, 1, 0, 1, 0, 1, 0, 1],
                                           [0, 1, 2, 0, 1, 2])
        triclustering.run_with_threads(nthreads=2)
        np.testing.assert_equal(triclustering.row_clusters,
                                [1, 1, 0, 3, 4, 0, 2, 2, 0, 0])
        np.testing.assert_equal(triclustering.col_clusters,
                                [0, 1, 0, 1, 1, 1, 0, 1])
        np.testing.assert_equal(triclustering.bnd_clusters,
                                [1, 0, 2, 0, 1, 0])
        assert np.isclose(triclustering.error, -69712.35398188536)

    def test_nruns_completed(self, triclustering):
        triclustering.run_with_threads(nthreads=1)
        assert triclustering.nruns_completed == 10
        triclustering.run_with_threads(nthreads=1)
        assert triclustering.nruns_completed == 20
