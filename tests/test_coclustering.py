import pytest

import numpy as np
import dask.array as da

from cgc.coclustering import Coclustering


@pytest.fixture
def coclustering():
    np.random.seed(1234)
    m, n = 10, 8
    ncl_row, ncl_col = 5, 2
    Z = np.random.randint(100, size=(m, n)).astype('float64')
    return Coclustering(Z, nclusters_row=ncl_row, nclusters_col=ncl_col,
                        conv_threshold=1.e-5, max_iterations=100, nruns=10,
                        epsilon=1.e-8)


class TestCoclustering:
    def test_check_initial_assignments(self, coclustering):
        assert coclustering.row_clusters is None
        assert coclustering.col_clusters is None

    def test_run_with_threads(self, coclustering):
        coclustering.run_with_threads(nthreads=2,
                                      row_clusters=[0, 1, 2, 3, 4,
                                                    0, 1, 2, 3, 4],
                                      col_clusters=[0, 1, 0, 1, 0, 1, 0, 1])
        np.testing.assert_equal(coclustering.row_clusters,
                                [3, 0, 1, 4, 0, 2, 2, 2, 3, 4])
        np.testing.assert_equal(coclustering.col_clusters,
                                [0, 0, 0, 1, 0, 0, 1, 1])
        assert np.isclose(coclustering.error, -11554.1406004284)

    def test_dask_runs_memory(self, client, coclustering):
        coclustering.run_with_dask(low_memory=True,
                                   row_clusters=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                                   col_clusters=[0, 1, 0, 1, 0, 1, 0, 1])
        np.testing.assert_equal(coclustering.row_clusters,
                                [3, 0, 1, 4, 0, 2, 2, 2, 3, 4])
        np.testing.assert_equal(coclustering.col_clusters,
                                [0, 0, 0, 1, 0, 0, 1, 1])
        assert np.isclose(coclustering.error, -11554.1406004284)

    def test_dask_runs_performance(self, client, coclustering):
        coclustering.run_with_dask(client=client, low_memory=False,
                                   row_clusters=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
                                   col_clusters=[0, 1, 0, 1, 0, 1, 0, 1])
        np.testing.assert_equal(coclustering.row_clusters,
                                [3, 0, 1, 4, 0, 2, 2, 2, 3, 4])
        np.testing.assert_equal(coclustering.col_clusters,
                                [0, 0, 0, 1, 0, 0, 1, 1])
        assert np.isclose(coclustering.error, -11554.1406004284)

    def test_nruns_completed_threads(self, coclustering):
        coclustering.run_with_threads(nthreads=1)
        assert coclustering.nruns_completed == 10
        coclustering.run_with_threads(nthreads=1)
        assert coclustering.nruns_completed == 10

    def test_nruns_completed_dask(self, client, coclustering):
        coclustering.run_with_dask(client)
        assert coclustering.nruns_completed == 10
        coclustering.run_with_dask(client, low_memory=True)
        assert coclustering.nruns_completed == 10
