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
    k_range = range(1, 3)
    kmean_max_iter = 2
    km = Kmeans(
        Z=Z,
        row_clusters=row_clusters,
        col_clusters=col_clusters,
        n_row_clusters=n_row_cluster,
        n_col_clusters=n_col_cluster,
        k_range=k_range,
        kmean_max_iter=kmean_max_iter)
    return km


class TestKmeans(unittest.TestCase):
    def test_kmean_force_n_clusters(self):
        Z = np.arange(20).reshape((5, 4))
        row_clusters = np.array([0, 0, 1, 1, 1])
        col_clusters = np.array([0, 1, 1, 2])
        n_row_cluster, n_col_cluster = 1, 1
        k_range = range(1, 3)
        kmean_max_iter = 100
        km = Kmeans(
            Z=Z,
            row_clusters=row_clusters,
            col_clusters=col_clusters,
            n_row_clusters=n_row_cluster,
            n_col_clusters=n_col_cluster,
            k_range=k_range,
            kmean_max_iter=kmean_max_iter)
        self.assertEqual(2, km.n_row_clusters)
        self.assertEqual(3, km.n_col_clusters)

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
