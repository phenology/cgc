import unittest
import numpy as np

from geoclustering.kmeans import Kmeans


def initialize_kmean():
    """
        Z:
        [[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11],
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
    k, l = 3, 2
    kmean_n_clusters, kmean_max_iter = 2, 100
    km = Kmeans(
        Z=Z,
        row_clusters=row_clusters,
        col_clusters=col_clusters,
        n_row_clusters=k,
        n_col_clusters=l,
        kmean_n_clusters=kmean_n_clusters,
        kmean_max_iter=kmean_max_iter)
    return km


class TestKmeans(unittest.TestCase):
    def test_kmean_force_n_clusters(self):
        Z = np.arange(20).reshape((5, 4))
        row_clusters = np.array([0, 0, 1, 1, 1])
        col_clusters = np.array([0, 1, 1, 2])
        k, l = 1, 1
        kmean_n_clusters, kmean_max_iter = 2, 100
        km = Kmeans(
            Z=Z,
            row_clusters=row_clusters,
            col_clusters=col_clusters,
            n_row_clusters=k,
            n_col_clusters=l,
            kmean_n_clusters=kmean_n_clusters,
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
        self.assertEqual((3, 2), km.cl_mean_centroids.shape)

    def test_centroids_nan(self):
        km = initialize_kmean()
        km.compute()
        self.assertTrue(all(np.isnan(km.cl_mean_centroids[2, :])))
