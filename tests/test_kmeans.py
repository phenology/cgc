import unittest
import numpy as np

from cgc.kmeans import Kmeans


def initialize_kmean():
    """
    Z:
        [[0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [16, 17, 18, 19]]

    cluster index:
        [[(0,0), (0,0), (0,1), (0,1)],
        [(0,0), (0,0), (0,1), (0,1)],
        [(1,0), (1,0), (1,1), (1,1)],
        [(1,0), (1,0), (1,1), (1,1)],
        [(1,0), (1,0), (1,1), (1,1)]]
    """
    Z = np.arange(20).reshape((5, 4))
    row_clusters = np.array([0, 0, 1, 1, 1])
    col_clusters = np.array([0, 0, 1, 1])
    n_row_cluster, n_col_cluster = 3, 2
    k_range = range(2, 4)
    kmean_max_iter = 2
    km = Kmeans(Z=Z,
                row_clusters=row_clusters,
                col_clusters=col_clusters,
                n_row_clusters=n_row_cluster,
                n_col_clusters=n_col_cluster,
                k_range=k_range,
                kmean_max_iter=kmean_max_iter)
    return km


class TestKmeans(unittest.TestCase):
    def test_kvalues_exceed_number_of_coclusters(self):
        with self.assertRaises(ValueError):
            Kmeans(
                Z=np.random.random((6, 4)),
                row_clusters=[0, 0, 1, 1, 2, 2],
                col_clusters=[0, 0, 1, 1],
                n_row_clusters=3,
                n_col_clusters=2,
                k_range=range(1, 8),
            )

    def test_kvalues_exceed_number_of_coclusters_populated(self):
        with self.assertRaises(ValueError):
            Kmeans(
                Z=np.random.random((6, 4)),
                row_clusters=[0, 0, 1, 1, 2, 2],
                col_clusters=[0, 0, 1, 1],
                n_row_clusters=4,
                n_col_clusters=2,
                k_range=range(1, 8),
            )

    def test_statistic_mesures_mean(self):
        km = initialize_kmean()
        km.compute()
        self.assertTrue(
            all(np.array([2.5, 4.5, 12.5, 14.5]) ==
                km.stat_measures[:, 0]))  # First colummn is mean

    def test_statistic_centroids_shape(self):
        km = initialize_kmean()
        km.compute()
        self.assertEqual((3, 2), km.results.cl_mean_centroids.shape)

    def test_centroids_nan(self):
        km = initialize_kmean()
        km.compute()
        self.assertTrue(all(np.isnan(km.results.cl_mean_centroids[2, :])))

    def test_kvalue_does_not_depend_on_krange_order(self):
        # 4 co-clusters, 2 clusters
        Z = np.array([
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0],
        ])
        row_cluseters = np.array([0, 0, 1, 1])
        col_clusters = np.array([0, 0, 0, 1, 1])
        km = Kmeans(Z=Z,
                    row_clusters=row_cluseters,
                    col_clusters=col_clusters,
                    n_row_clusters=2,
                    n_col_clusters=2,
                    k_range=range(2, 5))
        res = km.compute()
        self.assertEqual(res.k_value, 2)
        km = Kmeans(Z=Z,
                    row_clusters=row_cluseters,
                    col_clusters=col_clusters,
                    n_row_clusters=2,
                    n_col_clusters=2,
                    k_range=range(4, 1, -1))
        res = km.compute()
        self.assertEqual(res.k_value, 2)
