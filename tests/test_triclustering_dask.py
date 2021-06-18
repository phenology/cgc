import numpy as np
import dask.array as da

from cgc import triclustering_dask
from cgc.triclustering_dask import triclustering


class TestDistance:
    def test(self):
        d = 3
        m = 4
        n = 5
        k = 2  # number of band clusters
        np.random.seed(1234)
        Z = np.random.randint(100, size=(d, m, n)).astype('float64')
        Y = np.random.randint(100, size=(k, m, n)).astype('float64')
        Y[0, 3, 1] = 0.  # setting value to zero should not give infinite
        epsilon = 1.e-8
        distance = triclustering_dask._distance(Z, Y, epsilon)
        assert distance.shape == (d, k)
        assert ~np.any(np.isinf(distance))
        target = [
            [-1136.1376816, -2094.30771188],
            [-542.49375152, -2152.56590639],
            [-2074.05605694, -1505.49646914]
        ]
        assert np.allclose(distance, target)


class TestInitializeClusters:
    def test_all_points_are_assigned(self):
        m = 10
        k = 3
        clusters = triclustering_dask._initialize_clusters(m, k)
        assert set(clusters.compute().tolist()) == {i for i in range(k)}

    def test_all_clusters_are_initialized(self):
        # if m == k, all clusters should have initial occupation one
        m = 10
        k = 10
        clusters = triclustering_dask._initialize_clusters(m, k)
        assert sorted(clusters) == [i for i in range(k)]

    def test_more_clusters_than_elements(self):
        # only the first m clusters should be initialized
        m = 10
        k = 20
        clusters = triclustering_dask._initialize_clusters(m, k)
        assert set(clusters.compute().tolist()) == {i for i in range(m)}


class TestTriclustering:
    def test_small_matrix(self, client):
        np.random.seed(1234)
        Z = np.random.permutation(da.arange(60)).reshape((3, 4, 5))
        ncl_row = 3
        ncl_col = 4
        ncl_bnd = 2
        conv, niterations, row_cl, col_cl, bnd_cl, error = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-5, 100, 1.e-8,
            row_clusters_init=[1, 2, 0, 1],
            col_clusters_init=[1, 0, 3, 2, 0],
            bnd_clusters_init=[1, 0, 0]
        )
        assert conv
        assert niterations == 2
        np.testing.assert_array_equal(row_cl,
                                      np.array([1, 2, 0, 1]))
        np.testing.assert_array_equal(col_cl,
                                      np.array([1, 0, 3, 2, 0]))
        np.testing.assert_array_equal(bnd_cl,
                                      np.array([1, 0, 0]))
        np.testing.assert_almost_equal(error, -4330.426306789841)

    def test_bigger_matrix(self, client):
        np.random.seed(1234)
        nel_row = 20
        nel_col = 15
        nel_bnd = 10
        Z = np.random.randint(100, size=(nel_bnd, nel_row, nel_col))
        Z = Z.astype('float64')
        ncl_row = 5
        ncl_col = 6
        ncl_bnd = 3
        _, _, row_cl, col_cl, bnd_cl, _ = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-5, 100, 1.e-8,
            row_clusters_init=np.mod(np.arange(nel_row), ncl_row),
            col_clusters_init=np.mod(np.arange(nel_col), ncl_col),
            bnd_clusters_init=np.mod(np.arange(nel_bnd), ncl_bnd)
        )
        np.testing.assert_array_equal(np.sort(np.unique(row_cl)),
                                      np.arange(ncl_row))
        np.testing.assert_array_equal(np.sort(np.unique(col_cl)),
                                      np.arange(ncl_col))
        np.testing.assert_array_equal(np.sort(np.unique(bnd_cl)),
                                      np.arange(ncl_bnd))

    def test_as_many_clusters_as_elements(self, client):
        # it should immediately converge (2 iterations)
        ncl_row = 8
        ncl_col = 7
        ncl_bnd = 6
        Z = np.random.randint(100, size=(ncl_bnd, ncl_row, ncl_col))
        Z = Z.astype('float64')
        conv, niterations, _, _, _, _ = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-5, 100, 1.e-8,
            row_clusters_init=np.mod(np.arange(ncl_row), ncl_row),
            col_clusters_init=np.mod(np.arange(ncl_col), ncl_col),
            bnd_clusters_init=np.mod(np.arange(ncl_bnd), ncl_bnd)
        )
        assert conv
        assert niterations == 2

    def test_constant_row_matrix(self, client):
        # should give one cluster in rows
        Z = np.repeat(np.arange(135).reshape((9, 15)), 12, axis=0)
        Z = Z.reshape((9, 12, 15))
        ncl_row = 12
        ncl_col = 15
        ncl_bnd = 9
        _, _, row_cl, col_cl, bnd_cl, _ = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-5, 100, 1.e-8,
            row_clusters_init=np.mod(np.arange(12), ncl_row),
            col_clusters_init=np.mod(np.arange(15), ncl_col),
            bnd_clusters_init=np.mod(np.arange(9), ncl_bnd)
        )
        assert np.unique(row_cl.compute()).size == 1
        assert np.unique(col_cl.compute()).size == ncl_col
        assert np.unique(bnd_cl.compute()).size == ncl_bnd

    def test_constant_col_matrix(self, client):
        # should give one cluster in columns
        Z = np.repeat(np.arange(108), 15).reshape((9, 12, 15))
        ncl_row = 12
        ncl_col = 15
        ncl_bnd = 9
        _, _, row_cl, col_cl, bnd_cl, _ = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-5, 100, 1.e-8,
            row_clusters_init=np.mod(np.arange(12), ncl_row),
            col_clusters_init=np.mod(np.arange(15), ncl_col),
            bnd_clusters_init=np.mod(np.arange(9), ncl_bnd)
        )
        assert np.unique(row_cl.compute()).size == ncl_row
        assert np.unique(col_cl.compute()).size == 1
        assert np.unique(bnd_cl.compute()).size == ncl_bnd

    def test_constant_bnd_matrix(self, client):
        # should give one cluster in bands
        Z = np.tile(np.arange(180).reshape((12, 15)), (9, 1, 1))
        ncl_row = 12
        ncl_col = 15
        ncl_bnd = 9
        _, _, row_cl, col_cl, bnd_cl, _ = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-5, 100, 1.e-8,
            row_clusters_init=np.mod(np.arange(12), ncl_row),
            col_clusters_init=np.mod(np.arange(15), ncl_col),
            bnd_clusters_init=np.mod(np.arange(9), ncl_bnd)
        )
        assert np.unique(row_cl.compute()).size == ncl_row
        assert np.unique(col_cl.compute()).size == ncl_col
        assert np.unique(bnd_cl.compute()).size == 1

    def test_constant_rows_and_cols_matrix(self, client):
        Z = np.stack([np.full((12, 15), i) for i in range(1, 10)])
        ncl_row = 12
        ncl_col = 15
        ncl_bnd = 9
        _, _, row_cl, col_cl, bnd_cl, _ = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-5, 100, 1.e-8,
            row_clusters_init=np.mod(np.arange(12), ncl_row),
            col_clusters_init=np.mod(np.arange(15), ncl_col),
            bnd_clusters_init=np.mod(np.arange(9), ncl_bnd)
        )
        assert np.unique(row_cl.compute()).size == 1
        assert np.unique(col_cl.compute()).size == 1
        assert np.unique(bnd_cl.compute()).size == ncl_bnd

    def test_constant_rows_and_bnds_matrix(self, client):
        Z = np.stack([np.full((9, 12), i) for i in range(1, 16)])
        Z = Z.transpose((1, 2, 0))
        ncl_row = 12
        ncl_col = 15
        ncl_bnd = 9
        _, _, row_cl, col_cl, bnd_cl, _ = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-5, 100, 1.e-8,
            row_clusters_init=np.mod(np.arange(12), ncl_row),
            col_clusters_init=np.mod(np.arange(15), ncl_col),
            bnd_clusters_init=np.mod(np.arange(9), ncl_bnd)
        )
        assert np.unique(row_cl.compute()).size == 1
        assert np.unique(col_cl.compute()).size == ncl_col
        assert np.unique(bnd_cl.compute()).size == 1

    def test_constant_cols_and_bnds_matrix(self, client):
        Z = np.stack([np.full((9, 15), i) for i in range(1, 13)])
        Z = Z.transpose((1, 0, 2))
        ncl_row = 12
        ncl_col = 15
        ncl_bnd = 9
        _, _, row_cl, col_cl, bnd_cl, _ = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-20, 100, 1.e-8,
            row_clusters_init=np.mod(np.arange(12), ncl_row),
            col_clusters_init=np.mod(np.arange(15), ncl_col),
            bnd_clusters_init=np.mod(np.arange(9), ncl_bnd)
        )
        assert np.unique(row_cl.compute()).size == ncl_row
        assert np.unique(col_cl.compute()).size == 1
        assert np.unique(bnd_cl.compute()).size == 1

    def test_zero_matrix(self, client):
        # special case for the error - and no nan/inf
        Z = np.zeros((8, 9, 10))
        ncl_row = 9
        ncl_col = 10
        ncl_bnd = 8
        epsilon = 1.e-6
        _, _, _, _, _, error = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-5, 100, epsilon,
            row_clusters_init=np.mod(np.arange(9), ncl_row),
            col_clusters_init=np.mod(np.arange(10), ncl_col),
            bnd_clusters_init=np.mod(np.arange(8), ncl_bnd)
        )
        assert np.isclose(error, Z.size*epsilon)
