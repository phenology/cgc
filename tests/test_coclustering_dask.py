import numpy as np
import dask.array as da

from cgc import coclustering_dask
from cgc.coclustering_dask import coclustering


class TestDistance:
    def test(self):
        m = 4
        n = 5
        k = 2
        da.random.seed(1234)
        Z = da.random.randint(100, size=(m, n)).astype('float64')
        X = da.ones((m, n))
        Y = da.random.randint(2, size=(n, k)).astype('float64')
        epsilon = 1.e-8
        d = coclustering_dask._distance(Z, X, Y, epsilon)
        assert isinstance(d, da.core.Array)
        assert d.shape == (m, k)
        assert ~np.any(np.isinf(d))


class TestInitializeClusters:
    def test_return_dask_array(self):
        m = 10
        k = 3
        clusters = coclustering_dask._initialize_clusters(m, k)
        assert isinstance(clusters, da.core.Array)

    def test_all_points_are_assigned(self):
        m = 10
        k = 3
        clusters = coclustering_dask._initialize_clusters(m, k)
        assert set(clusters.compute().tolist()) == {i for i in range(k)}

    def test_all_clusters_are_initialized(self):
        # if m == k, all clusters should have initial occupation one
        m = 10
        k = 10
        clusters = coclustering_dask._initialize_clusters(m, k)
        assert sorted(clusters) == [i for i in range(k)]

    def test_more_clusters_than_elements(self):
        # only the first m clusters should be initialized
        m = 10
        k = 20
        clusters = coclustering_dask._initialize_clusters(m, k)
        assert set(clusters.compute().tolist()) == {i for i in range(m)}


class TestCoclustering:
    """
    Test coclustering function in Dask. We make use of the client pytest
    fixture from distributed.utils_test, which we import in `conftest.py`.
    """
    def test_return_dask_array(self, client):
        Z = da.random.permutation(da.arange(4)).reshape(2, 2)
        ncl_row = 2
        ncl_col = 2
        da.random.seed(1234)
        _, _, row_cl, col_cl, _ = coclustering(
            Z, ncl_row, ncl_col, 1.e-5, 1, 1.e-8
        )
        assert isinstance(row_cl, da.core.Array)
        assert isinstance(col_cl, da.core.Array)

    def test_small_matrix(self, client):
        da.random.seed(1234)
        Z = da.random.permutation(da.arange(12)).reshape(3, 4)
        ncl_row = 2
        ncl_col = 3
        conv, niterations, row_cl, col_cl, error = coclustering(
            Z, ncl_row, ncl_col, 1.e-5, 100, 1.e-8
        )
        assert conv
        assert niterations == 3
        np.testing.assert_array_equal(row_cl.compute(),
                                      np.array([1, 0, 0]))
        np.testing.assert_array_equal(col_cl.compute(),
                                      np.array([1, 1, 2, 0]))
        np.testing.assert_almost_equal(error, -56.457907947376775)

    def test_bigger_matrix(self, client):
        Z = da.random.randint(100, size=(20, 15)).astype('float64')
        ncl_row = 5
        ncl_col = 6
        da.random.seed(1234)
        _, _, row_cl, col_cl, _ = coclustering(
            Z, ncl_row, ncl_col, 1.e-5, 100, 1.e-8
        )
        np.testing.assert_array_equal(np.sort(np.unique(row_cl.compute())),
                                      np.arange(ncl_row))
        np.testing.assert_array_equal(np.sort(np.unique(col_cl.compute())),
                                      np.arange(ncl_col))

    def test_as_many_clusters_as_elements(self, client):
        # it should immediately converge (2 iterations)
        ncl_row = 8
        ncl_col = 7
        da.random.seed(1234)
        Z = da.random.randint(100, size=(ncl_row, ncl_col)).astype('float64')
        conv, niterations, _, _, _ = coclustering(
            Z, ncl_row, ncl_col, 1.e-5, 100, 1.e-8
        )
        assert conv
        assert niterations == 2

    def test_constant_col_matrix(self, client):
        # should give one cluster in rows
        Z = np.tile(np.arange(7), (8, 1))
        Z = da.from_array(Z)
        ncl_row = 3
        ncl_col = 4
        da.random.seed(1234)
        _, _, row_cl, col_cl, _ = coclustering(
            Z, ncl_row, ncl_col, 1.e-5, 100, 1.e-8
        )
        print(col_cl.compute())
        assert np.unique(row_cl.compute()).size == 1
        assert np.unique(col_cl.compute()).size == ncl_col

    def test_constant_row_matrix(self, client):
        # should give one cluster in columns
        Z = np.repeat(np.arange(8), 7).reshape(8, 7)
        Z = da.from_array(Z)
        ncl_row = 3
        ncl_col = 4
        da.random.seed(1234)
        _, _, row_cl, col_cl, _ = coclustering(
            Z, ncl_row, ncl_col, 1.e-5, 100, 1.e-8
        )
        assert np.unique(row_cl.compute()).size == ncl_row
        assert np.unique(col_cl.compute()).size == 1

    def test_zero_matrix(self, client):
        # special case for the error - and no nan/inf
        Z = np.zeros((8, 7))
        Z = da.from_array(Z)
        ncl_row = 3
        ncl_col = 4
        epsilon = 1.e-6
        da.random.seed(1234)
        _, _, _, _, error = coclustering(
            Z, ncl_row, ncl_col, 1.e-5, 100, epsilon
        )
        assert np.isclose(error, Z.size*epsilon)
