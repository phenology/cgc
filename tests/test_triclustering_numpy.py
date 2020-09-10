import numpy as np

from cgc import triclustering_numpy
from cgc.triclustering_numpy import triclustering


class TestDistance:
    def test(self):
        m = 4
        n = 5
        k = 2
        np.random.seed(1234)
        Z = np.random.randint(100, size=(m, n)).astype('float64')
        X = np.ones((m, n))
        Y = np.random.randint(100, size=(n, k)).astype('float64')
        Y[3, 1] = 0.  # setting value to zero should not give infinite
        epsilon = 1.e-8
        d = triclustering_numpy._distance(Z, X, Y, epsilon)
        assert d.shape == (m, k)
        assert ~np.any(np.isinf(d))


class TestInitializeClusters:
    def test_all_points_are_assigned(self):
        m = 10
        k = 3
        clusters = triclustering_numpy._initialize_clusters(m, k)
        assert set(clusters.tolist()) == {i for i in range(k)}

    def test_all_clusters_are_initialized(self):
        # if m == k, all clusters should have initial occupation one
        m = 10
        k = 10
        clusters = triclustering_numpy._initialize_clusters(m, k)
        assert sorted(clusters) == [i for i in range(k)]

    def test_more_clusters_than_elements(self):
        # only the first m clusters should be initialized
        m = 10
        k = 20
        clusters = triclustering_numpy._initialize_clusters(m, k)
        assert set(clusters.tolist()) == {i for i in range(m)}


class TestTriclustering:
    def test_small_matrix(self):
        np.random.seed(1234)
        Z = np.random.permutation(np.arange(60)).reshape(3, 4, 5)
        ncl_row = 3
        ncl_col = 4
        ncl_bnd = 2
        conv, niterations, row_cl, col_cl, bnd_cl, error = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-5, 100, 1.e-8,
            row_clusters_init=[0, 2, 0, 1],
            col_clusters_init=[1, 0, 3, 2, 0],
            bnd_clusters_init=[1, 0, 0]
        )
        assert conv
        assert niterations == 2
        np.testing.assert_array_equal(row_cl,
                                      np.array([0, 2, 0, 1]))
        np.testing.assert_array_equal(col_cl,
                                      np.array([1, 0, 3, 2, 0]))
        np.testing.assert_array_equal(bnd_cl,
                                      np.array([1, 0, 0]))
        np.testing.assert_almost_equal(error, -4249.966724020571)

    def test_bigger_matrix(self):
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

    def test_as_many_clusters_as_elements(self):
        # it should immediately converge (2 iterations)
        ncl_row = 8
        ncl_col = 7
        ncl_bnd = 6
        Z = np.random.randint(100, size=(ncl_bnd, ncl_row, ncl_col))
        Z = Z.astype('float64')
        conv, niterations, _, _, _, _ = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-5, 100, 1.e-8
        )
        assert conv
        assert niterations == 2

    def test_constant_col_matrix(self):
        # should give one cluster in rows
        Z = np.repeat(np.arange(72).reshape((8, 9)), 10, axis=0)
        Z = Z.reshape((8, 10, 9))
        ncl_row = 3
        ncl_col = 3
        ncl_bnd = 3
        _, _, row_cl, col_cl, bnd_cl, _ = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-5, 100, 1.e-8
        )
        assert np.unique(row_cl).size == 1
        assert np.unique(col_cl).size == ncl_col
        assert np.unique(bnd_cl).size == ncl_bnd

    def test_constant_row_matrix(self):
        # should give one cluster in columns
        Z = np.repeat(np.arange(80), 9).reshape(8, 10, 9)
        ncl_row = 3
        ncl_col = 3
        ncl_bnd = 3
        _, _, row_cl, col_cl, bnd_cl, _ = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-5, 100, 1.e-8
        )
        assert np.unique(row_cl).size == ncl_row
        assert np.unique(col_cl).size == 1
        assert np.unique(bnd_cl).size == ncl_bnd

    def test_constant_bnd_matrix(self):
        # should give one cluster in bands
        Z = np.tile(np.arange(90).reshape((9, 10)), (8, 1, 1))
        ncl_row = 3
        ncl_col = 3
        ncl_bnd = 3
        _, _, row_cl, col_cl, bnd_cl, _ = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-5, 100, 1.e-8
        )
        assert np.unique(row_cl).size == ncl_row
        assert np.unique(col_cl).size == ncl_col
        assert np.unique(bnd_cl).size == 1

    def test_zero_matrix(self):
        # special case for the error - and no nan/inf
        Z = np.zeros((8, 9, 10))
        ncl_row = 3
        ncl_col = 3
        ncl_bnd = 3
        epsilon = 1.e-6
        _, _, _, _, _, error = triclustering(
            Z, ncl_row, ncl_col, ncl_bnd, 1.e-5, 100, epsilon
        )
        assert np.isclose(error, Z.size*epsilon)
