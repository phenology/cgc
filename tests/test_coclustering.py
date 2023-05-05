import os
import tempfile

import numpy as np
import pytest

from cgc.coclustering import Coclustering


@pytest.fixture
def coclustering():
    np.random.seed(1234)
    m, n = 10, 8
    ncl_row, ncl_col = 5, 2
    Z = np.random.randint(100, size=(m, n)).astype('float64')
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Coclustering(
            Z,
            nclusters_row=ncl_row,
            nclusters_col=ncl_col,
            conv_threshold=1.e-5,
            max_iterations=100,
            nruns=1,
            row_clusters_init=[0, 1, 2, 3, 4, 0, 1, 2, 3, 4],
            col_clusters_init=[0, 1, 0, 1, 0, 1, 0, 1],
            output_filename=os.path.join(tmpdir, "results.json")
        )


class TestCoclustering:
    def test_check_initial_assignments(self, coclustering):
        assert coclustering.results.row_clusters is None
        assert coclustering.results.col_clusters is None

    def test_run_with_threads(self, coclustering):
        coclustering.run_with_threads(nthreads=2)
        assert os.path.isfile(coclustering.output_filename)
        np.testing.assert_equal(coclustering.results.row_clusters,
                                [0, 0, 2, 4, 1, 0, 1, 2, 3, 4])
        np.testing.assert_equal(coclustering.results.col_clusters,
                                [0, 1, 0, 0, 0, 1, 0, 1])
        assert np.isclose(coclustering.results.error, -11503.89447245418)

    def test_run_with_threads_lowmem(self, coclustering):
        coclustering.run_with_threads(nthreads=2, low_memory=True)
        assert os.path.isfile(coclustering.output_filename)
        np.testing.assert_equal(coclustering.results.row_clusters,
                                [0, 0, 2, 4, 1, 0, 1, 2, 3, 4])
        np.testing.assert_equal(coclustering.results.col_clusters,
                                [0, 1, 0, 0, 0, 1, 0, 1])
        assert np.isclose(coclustering.results.error, -11503.89447245418)

    def test_dask_runs_memory(self, client, coclustering):
        coclustering.run_with_dask(client=client, low_memory=True)
        assert os.path.isfile(coclustering.output_filename)
        np.testing.assert_equal(coclustering.results.row_clusters,
                                [0, 0, 2, 4, 1, 0, 1, 2, 3, 4])
        np.testing.assert_equal(coclustering.results.col_clusters,
                                [0, 1, 0, 0, 0, 1, 0, 1])
        assert np.isclose(coclustering.results.error, -11503.89447245418)

    def test_dask_runs_performance(self, client, coclustering):
        coclustering.run_with_dask(client=client, low_memory=False)
        assert os.path.isfile(coclustering.output_filename)
        np.testing.assert_equal(coclustering.results.row_clusters,
                                [0, 0, 2, 4, 1, 0, 1, 2, 3, 4])
        np.testing.assert_equal(coclustering.results.col_clusters,
                                [0, 1, 0, 0, 0, 1, 0, 1])
        assert np.isclose(coclustering.results.error, -11503.89447245418)

    def test_nruns_completed_threads(self, coclustering):
        coclustering.run_with_threads(nthreads=1)
        assert coclustering.results.nruns_completed == 1
        coclustering.run_with_threads(nthreads=1)
        assert coclustering.results.nruns_completed == 2

    def test_nruns_completed_dask(self, client, coclustering):
        coclustering.run_with_dask(client)
        assert coclustering.results.nruns_completed == 1
        coclustering.run_with_dask(client, low_memory=True)
        assert coclustering.results.nruns_completed == 2
