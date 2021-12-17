import numpy as np

from cgc import coclustering_numpy
from cgc.coclustering_numpy import coclustering


class TestDistance:
    def test_distance_basic(self):
        # original matrix can contain zeros..
        Z = np.arange(0, 40).reshape(8, 5).astype('float64')
        row_clusters = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=int)
        col_clusters = np.array([0, 0, 0, 1, 1], dtype=int)
        # ..not the cluster average
        cc = np.arange(1, 9).reshape(4, 2)
        R = np.eye(4, dtype=bool)[row_clusters]
        C = np.eye(2, dtype=bool)[col_clusters]
        _, d_row = coclustering_numpy._min_dist(Z, C, cc.T)
        _, d_col = coclustering_numpy._min_dist(Z.T, R, cc)
        assert d_row.size == 8
        assert d_col.size == 5
        assert np.isfinite(d_row).all()
        assert np.isfinite(d_col).all()


class TestInitializeClusters:
    def test_all_points_are_assigned(self):
        m = 10
        k = 3
        clusters = coclustering_numpy._initialize_clusters(m, k)
        assert set(clusters.tolist()) == {i for i in range(k)}

    def test_all_clusters_are_initialized(self):
        # if m == k, all clusters should have initial occupation one
        m = 10
        k = 10
        clusters = coclustering_numpy._initialize_clusters(m, k)
        assert sorted(clusters) == [i for i in range(k)]

    def test_more_clusters_than_elements(self):
        # only the first m clusters should be initialized
        m = 10
        k = 20
        clusters = coclustering_numpy._initialize_clusters(m, k)
        assert set(clusters.tolist()) == {i for i in range(m)}


class TestCoclustering:
    def test_small_matrix(self):
        np.random.seed(1234)
        Z = np.random.permutation(np.arange(12)).reshape(3, 4)
        Z = Z.astype('float64')
        ncl_row = 2
        ncl_col = 3
        conv, niterations, row_cl, col_cl, error = coclustering(
            Z, ncl_row, ncl_col, 1.e-5, 100)
        assert conv
        assert niterations == 3
        np.testing.assert_array_equal(row_cl, np.array([1, 0, 0]))
        np.testing.assert_array_equal(col_cl, np.array([1, 1, 2, 0]))
        np.testing.assert_almost_equal(error, -56.457907947376775)

    def test_bigger_matrix(self):
        np.random.seed(1234)
        Z = np.random.randint(100, size=(20, 15)).astype('float64')
        ncl_row = 5
        ncl_col = 6
        _, _, row_cl, col_cl, _ = coclustering(Z, ncl_row, ncl_col, 1.e-5, 100)
        np.testing.assert_array_equal(np.sort(np.unique(row_cl)),
                                      np.arange(ncl_row))
        np.testing.assert_array_equal(np.sort(np.unique(col_cl)),
                                      np.arange(ncl_col))

    def test_as_many_clusters_as_elements(self):
        # it should immediately converge (2 iterations)
        ncl_row = 8
        ncl_col = 7
        Z = np.arange(1, ncl_row * ncl_col + 1).reshape(ncl_row, ncl_col)
        Z = Z.astype(float)
        conv, niterations, _, _, e = coclustering(Z, ncl_row, ncl_col, 1.e-5,
                                                  100)
        assert conv
        assert niterations == 2
        assert np.isfinite(e)

    def test_constant_col_matrix(self):
        # should give one cluster in rows
        Z = np.tile(np.arange(1, 8), (8, 1)).astype(float)
        ncl_row = 3
        ncl_col = 7
        _, _, row_cl, col_cl, e = coclustering(Z, ncl_row, ncl_col, 1.e-5, 100)
        assert np.unique(row_cl).size == 1
        assert np.unique(col_cl).size == ncl_col
        assert np.isfinite(e)

    def test_constant_row_matrix(self):
        # should give one cluster in columns
        Z = np.repeat(np.arange(1, 9), 7).reshape(8, 7).astype(float)
        ncl_row = 8
        ncl_col = 4
        _, _, row_cl, col_cl, e = coclustering(Z, ncl_row, ncl_col, 1.e-5, 100)
        assert np.unique(row_cl).size == ncl_row
        assert np.unique(col_cl).size == 1
        assert np.isfinite(e)

    def test_constant_matrix(self):
        # should give one cluster in column and rows
        Z = np.ones((8, 7)) * 5
        ncl_row = 3
        ncl_col = 4
        _, _, row_cl, col_cl, e = coclustering(Z, ncl_row, ncl_col, 1.e-5, 100)
        assert np.unique(row_cl).size == 1
        assert np.unique(col_cl).size == 1
        assert np.isfinite(e)


class TestLowMemoryNumba:
    def test_distance_lowmem_numba(self):
        # original matrix can contain zeros..
        Z = np.arange(0, 40).reshape(8, 5).astype('float64')
        row_clusters = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=int)
        col_clusters = np.array([0, 0, 0, 1, 1], dtype=int)
        # ..not the cluster average
        cc = np.arange(1, 9).reshape(4, 2)
        R = np.eye(4, dtype=bool)[row_clusters]
        C = np.eye(2, dtype=bool)[col_clusters]
        row_cl, d_row = coclustering_numpy._min_dist(Z, C, cc.T)
        col_cl, d_col = coclustering_numpy._min_dist(Z.T, R, cc)

        row_cl_numba, d_row_numba = coclustering_numpy._min_dist_numba(
            Z, col_clusters, np.arange(2), cc.T)
        col_cl_numba, d_col_numba = coclustering_numpy._min_dist_numba(
            Z.T, row_clusters, np.arange(4), cc)

        np.testing.assert_array_equal(row_cl, row_cl_numba)
        np.testing.assert_array_equal(col_cl, col_cl_numba)
        np.testing.assert_almost_equal(d_row, d_row_numba, 10)
        np.testing.assert_almost_equal(d_col, d_col_numba, 10)

    def test_dot_clustering_lowmem_numba(self):
        Z = np.arange(0, 40).reshape(8, 5).astype('float64')
        row_clusters = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=int)
        col_clusters = np.array([0, 0, 0, 1, 1], dtype=int)
        R = np.eye(4, dtype=int)[row_clusters]
        C = np.eye(2, dtype=int)[col_clusters]
        CoCavg = np.dot(np.dot(R.T, Z), C)
        CoCavg_numba = (coclustering_numpy._cluster_dot_numba(
            Z, row_clusters, col_clusters, np.arange(4), np.arange(2)))
        np.testing.assert_equal(CoCavg, CoCavg_numba)
