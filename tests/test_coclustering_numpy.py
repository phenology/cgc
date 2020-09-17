import numpy as np

from cgc import coclustering_numpy
from cgc.coclustering_numpy import coclustering


class TestDistance:
    def test_distance_basic(self):
        Z = np.arange(0, 40).reshape(8, 5).astype('float64')
        row_clusters = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
        col_clusters = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        cc = np.arange(1, 9).reshape(4, 2)
        np.random.seed(1234)
        R = np.eye(4, dtype=np.int32)[row_clusters]
        C = np.eye(2, dtype=np.int32)[col_clusters]
        R[1, :] = 0
        C[2, :] = 0
        CoCavg = coclustering_numpy._cluster_dot(Z, row_clusters, col_clusters,
                                                 4, 2)
        d_row = coclustering_numpy._min_dist(Z, C, CoCavg.T, 1.e-6)
        d_col = coclustering_numpy._min_dist(Z.T, R, CoCavg, 1.e-6)

        assert d_row[1].size == 8
        assert d_col[1].size == 5
        assert ~np.any(np.isinf(d_row[1]))
        assert ~np.any(np.isinf(d_col[1]))


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
        Z = np.random.permutation(np.arange(12)).reshape(3,
                                                         4).astype('float64')
        ncl_row = 2
        ncl_col = 3
        conv, niterations, row_cl, col_cl, error = coclustering(
            Z, ncl_row, ncl_col, 1.e-5, 100, 1.e-8)
        assert conv
        assert niterations == 3
        np.testing.assert_array_equal(row_cl, np.array([1, 0, 0]))
        np.testing.assert_array_equal(col_cl, np.array([1, 1, 2, 0]))
        np.testing.assert_almost_equal(error, -56.457907947376775)

    def test_bigger_matrix(self):
        Z = np.random.randint(100, size=(20, 15)).astype('float64')
        ncl_row = 5
        ncl_col = 6
        np.random.seed(1234)
        _, _, row_cl, col_cl, _ = coclustering(Z, ncl_row, ncl_col, 1.e-5, 100,
                                               1.e-8)
        np.testing.assert_array_equal(np.sort(np.unique(row_cl)),
                                      np.arange(ncl_row))
        np.testing.assert_array_equal(np.sort(np.unique(col_cl)),
                                      np.arange(ncl_col))

    def test_as_many_clusters_as_elements(self):
        # it should immediately converge (2 iterations)
        ncl_row = 8
        ncl_col = 7
        np.random.seed(1234)
        Z = np.random.randint(100, size=(ncl_row, ncl_col)).astype('float64')
        conv, niterations, _, _, _ = coclustering(Z, ncl_row, ncl_col, 1.e-5,
                                                  100, 1.e-8)
        assert conv
        assert niterations == 2

    def test_constant_col_matrix(self):
        # should give one cluster in rows
        Z = np.tile(np.arange(7), (8, 1)).astype('float64')
        ncl_row = 3
        ncl_col = 4
        np.random.seed(1234)
        _, _, row_cl, col_cl, _ = coclustering(Z, ncl_row, ncl_col, 1.e-5, 100,
                                               1.e-8)
        assert np.unique(row_cl).size == 1
        assert np.unique(col_cl).size == ncl_col

    def test_constant_row_matrix(self):
        # should give one cluster in columns
        Z = np.repeat(np.arange(8), 7).reshape(8, 7).astype('float64')
        ncl_row = 3
        ncl_col = 4
        np.random.seed(1234)
        _, _, row_cl, col_cl, _ = coclustering(Z, ncl_row, ncl_col, 1.e-5, 100,
                                               1.e-8)
        assert np.unique(row_cl).size == ncl_row
        assert np.unique(col_cl).size == 1

    def test_zero_matrix(self):
        # special case for the error - and no nan/inf
        Z = np.zeros((8, 7))
        ncl_row = 3
        ncl_col = 4
        epsilon = 1.e-6
        np.random.seed(1234)
        _, _, _, _, error = coclustering(Z, ncl_row, ncl_col, 1.e-5, 100,
                                         epsilon)
        assert np.isclose(error, Z.size * epsilon)


class TestLowMemory:
    def test_distance_lowmem(self):
        Z = np.arange(0, 40).reshape(8, 5).astype('float64')
        row_clusters = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
        col_clusters = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        cc = np.arange(1, 9).reshape(4, 2)
        np.random.seed(1234)
        R = np.eye(4, dtype=np.int32)[row_clusters]
        C = np.eye(2, dtype=np.int32)[col_clusters]
        CoCavg = coclustering_numpy._cluster_dot(Z, row_clusters, col_clusters,
                                                 4, 2)

        d_row = coclustering_numpy._min_dist(Z, C, CoCavg.T, 1.e-6)
        d_row_lowmem_numba = coclustering_numpy._min_dist_lowmem(
            Z, col_clusters, CoCavg.T, 1.e-6)

        d_col = coclustering_numpy._min_dist(Z.T, R, CoCavg, 1.e-6)
        d_col_lowmem_numba = coclustering_numpy._min_dist_lowmem(
            Z.T, row_clusters, CoCavg, 1.e-6)
        # Test almost equal because numpy sum lose precision
        np.testing.assert_almost_equal(d_row, d_row_lowmem_numba, 10)
        np.testing.assert_almost_equal(d_col, d_col_lowmem_numba, 10)

    def test_dot_clustering_lowmem(self):
        Z = np.arange(0, 40).reshape(8, 5).astype('float64')
        row_clusters = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
        col_clusters = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        R = np.eye(4, dtype=np.int32)[row_clusters]
        C = np.eye(2, dtype=np.int32)[col_clusters]
        CoCavg = (np.dot(np.dot(R.T, Z), C) + Z.mean() * 1.e-6) / (
            np.dot(np.dot(R.T, np.ones(Z.shape)), C) + 1.e-6)
        CoCavg_lowmem = (coclustering_numpy._cluster_dot(
            Z, row_clusters, col_clusters, 4, 2) + Z.mean() * 1.e-6) / (
                coclustering_numpy._cluster_dot(np.ones(Z.shape), row_clusters,
                                                col_clusters, 4, 2) + 1.e-6)
        np.testing.assert_equal(CoCavg, CoCavg_lowmem)


class TestLowMemoryNumba:
    def test_distance_lowmem_numba(self):
        Z = np.arange(0, 40).reshape(8, 5).astype('float64')
        row_clusters = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
        col_clusters = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        cc = np.arange(1, 9).reshape(4, 2)
        np.random.seed(1234)
        R = np.eye(4, dtype=np.int32)[row_clusters]
        C = np.eye(2, dtype=np.int32)[col_clusters]
        CoCavg = coclustering_numpy._cluster_dot(Z, row_clusters, col_clusters,
                                                 4, 2)

        d_row = coclustering_numpy._min_dist(Z, C, CoCavg.T, 1.e-6)
        d_row_lowmem_numba = coclustering_numpy._min_dist_numba(
            Z, col_clusters, CoCavg.T, 1.e-6)

        d_col = coclustering_numpy._min_dist(Z.T, R, CoCavg, 1.e-6)
        d_col_lowmem_numba = coclustering_numpy._min_dist_numba(
            Z.T, row_clusters, CoCavg, 1.e-6)
        # Test almost equal because numpy sum lose precision
        np.testing.assert_almost_equal(d_row, d_row_lowmem_numba, 10)
        np.testing.assert_almost_equal(d_col, d_col_lowmem_numba, 10)

    def test_dot_clustering_lowmem_numba(self):
        Z = np.arange(0, 40).reshape(8, 5).astype('float64')
        row_clusters = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)
        col_clusters = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        R = np.eye(4, dtype=np.int32)[row_clusters]
        C = np.eye(2, dtype=np.int32)[col_clusters]
        CoCavg = (np.dot(np.dot(R.T, Z), C) + Z.mean() * 1.e-6) / (
            np.dot(np.dot(R.T, np.ones(Z.shape)), C) + 1.e-6)
        CoCavg_lowmem_numba = (coclustering_numpy._cluster_dot_numba(
            Z, row_clusters, col_clusters, 4, 2) + Z.mean() * 1.e-6) / (
                coclustering_numpy._cluster_dot_numba(np.ones(
                    Z.shape), row_clusters, col_clusters, 4, 2) + 1.e-6)
        np.testing.assert_equal(CoCavg, CoCavg_lowmem_numba)
