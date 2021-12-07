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
                         max_iterations=100, nruns=1,
                         row_clusters_init=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                         col_clusters_init=[0, 1, 0, 1, 0, 1, 0, 1],
                         bnd_clusters_init=[0, 1, 2, 0, 1, 2])


class TestTriclustering:
    def test_check_initial_assignments(self, triclustering):
        assert triclustering.results.row_clusters is None
        assert triclustering.results.col_clusters is None
        assert triclustering.results.bnd_clusters is None

    def test_run_with_threads(self, triclustering):
        triclustering.run_with_threads(nthreads=2)
        np.testing.assert_equal(triclustering.results.row_clusters,
                                [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        np.testing.assert_equal(triclustering.results.col_clusters,
                                [0, 1, 0, 1, 0, 0, 1, 1])
        np.testing.assert_equal(triclustering.results.bnd_clusters,
                                [0, 1, 2, 0, 2, 2])
        assert np.isclose(triclustering.results.error, -70021.27155444604)

    def test_run_with_dask(self, client, triclustering):
        triclustering.run_with_dask(client)
        np.testing.assert_equal(triclustering.results.row_clusters,
                                [0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        np.testing.assert_equal(triclustering.results.col_clusters,
                                [0, 1, 0, 1, 0, 0, 1, 1])
        np.testing.assert_equal(triclustering.results.bnd_clusters,
                                [0, 1, 2, 0, 2, 2])
        assert np.isclose(triclustering.results.error, -70021.27155444604)

    def test_nruns_completed(self, triclustering):
        triclustering.run_with_threads(nthreads=1)
        assert triclustering.results.nruns_completed == 1
        triclustering.run_with_threads(nthreads=1)
        assert triclustering.results.nruns_completed == 2
