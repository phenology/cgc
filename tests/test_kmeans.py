from cgc.results import Results
import unittest
from unittest.runner import TextTestResult
import numpy as np

from cgc.kmeans import Kmeans


def init_cocluster():
    """
    Z:
        [[0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 3, 3],
        [2, 2, 3, 3],
        [2, 2, 3, 3]]

    cluster index:
        [[(0,0), (0,0), (0,1), (0,1)],
        [(0,0), (0,0), (0,1), (0,1)],
        [(1,0), (1,0), (1,1), (1,1)],
        [(1,0), (1,0), (1,1), (1,1)],
        [(1,0), (1,0), (1,1), (1,1)]]
    """
    Z = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3],
                  [2, 2, 3, 3]])
    row_clusters = np.array([0, 0, 1, 1, 1])
    col_clusters = np.array([0, 0, 1, 1])
    nrow_clusters, ncol_clusters = 3, 2  # 1 non populated row/col cluster
    clusters = [row_clusters, col_clusters]
    nclusters = [nrow_clusters, ncol_clusters]
    k_range = range(2, 4)
    kmean_max_iter = 2
    km = Kmeans(Z=Z,
                clusters=clusters,
                nclusters=nclusters,
                k_range=k_range,
                kmean_max_iter=kmean_max_iter)
    return km


def init_tricluster():
    Z1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3],
                   [2, 2, 3, 3]])
    Z = np.dstack((Z1, Z1, Z1 + 1, Z1 + 1))
    row_clusters = np.array([0, 0, 1, 1, 1])
    col_clusters = np.array([0, 0, 1, 1])
    band_clusters = np.array([0, 0, 1, 1])
    nrow_clusters, ncol_clusters, nband_clusters = 2, 2, 2
    clusters = [row_clusters, col_clusters, band_clusters]
    nclusters = [nrow_clusters, ncol_clusters, nband_clusters]
    k_range = range(2, 4)
    kmean_max_iter = 2
    km = Kmeans(Z=Z,
                clusters=clusters,
                nclusters=nclusters,
                k_range=k_range,
                kmean_max_iter=kmean_max_iter)
    return km


class TestKmeans(unittest.TestCase):
    def test_Z_and_cluster_shape_not_match(self):
        with self.assertRaises(ValueError):
            Kmeans(Z=np.random.random((5, 5)),
                   clusters=[[0, 0, 1, 1, 2], [0, 0, 1, 1]],
                   nclusters=[3, 2])

    def test_Z_and_cluster_dimension_not_match(self):
        with self.assertRaises(ValueError):
            Kmeans(Z=np.random.random((5, 4)),
                   clusters=[[0, 0, 1, 1, 2], [0, 0, 1, 1], [0, 0, 1, 1]],
                   nclusters=[3, 2, 2])

    def test_max_label_equal_ncluster(self):
        with self.assertRaises(ValueError):
            Kmeans(Z=np.random.random((5, 4, 4)),
                   clusters=[[0, 0, 1, 1, 2], [0, 0, 1, 1], [0, 0, 1, 1]],
                   nclusters=[3, 2, 1])

    def test_max_label_exceeds_ncluster(self):
        with self.assertRaises(ValueError):
            Kmeans(Z=np.random.random((5, 4, 4)),
                   clusters=[[0, 0, 1, 1, 2], [0, 0, 1, 1], [0, 0, 1, 1]],
                   nclusters=[2, 2, 2])

    def test_kvalues_exceed_number_of_coclusters(self):
        with self.assertRaises(ValueError):
            Kmeans(
                Z=np.random.random((6, 4)),
                clusters=[[0, 0, 1, 1, 2, 2], [0, 0, 1, 1]],
                nclusters=[3, 2],
                k_range=range(1, 8),
            )

    def test_kvalues_exceed_number_of_coclusters_populated(self):
        with self.assertRaises(ValueError):
            Kmeans(
                Z=np.random.random((6, 4)),
                clusters=[[0, 0, 1, 1, 2, 2], [0, 0, 1, 1]],
                nclusters=[4, 2],
                k_range=range(1, 8),
            )

    def test_statistic_coclustering(self):
        km = init_cocluster()
        km._compute_statistic_measures()
        results = np.array([[0., 0., 0., 0., 0., 0.], [1., 0., 1., 1., 1., 1.],
                            [2., 0., 2., 2., 2., 2.], [3., 0., 3., 3., 3.,
                                                       3.]])
        self.assertTrue(
            np.all(results == km.stat_measures))  # First colummn is mean

    def test_kmeam_labels_coclustering(self):
        km = init_cocluster()
        km.compute()
        labels = km.kmean_cluster.labels_
        self.assertTrue(labels.shape == (4, ))
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])

    def test_statistic_triclustering(self):
        km = init_tricluster()
        km._compute_statistic_measures()
        results = np.array([[0., 0., 0., 0., 0., 0.], [1., 0., 1., 1., 1., 1.],
                            [1., 0., 1., 1., 1., 1.], [2., 0., 2., 2., 2., 2.],
                            [2., 0., 2., 2., 2., 2.], [3., 0., 3., 3., 3., 3.],
                            [3., 0., 3., 3., 3., 3.], [4., 0., 4., 4., 4.,
                                                       4.]])
        self.assertTrue(
            np.all(results == km.stat_measures))  # First colummn is mean

    def test_kvalues_triclustering(self):
        km = init_tricluster()
        km.compute()
        self.assertEqual(km.results.k_value, 3)

    def test_statistic_centroids_shape(self):
        km = init_cocluster()
        km.compute()
        self.assertEqual((3, 2), km.results.cl_mean_centroids.shape)

    def test_centroids_nan(self):
        km = init_cocluster()
        km.compute()
        self.assertTrue(all(np.isnan(km.results.cl_mean_centroids[2, :])))

    def test_kvalue_does_not_depend_on_krange_order(self):
        # 4 co-clusters, 2 clusters
        Z = np.array([[1, 1, 1, 2, 2], [1, 1, 1, 2, 2], [2, 2, 2, 1, 1],
                      [2, 2, 2, 1, 1]])
        Z = Z + np.random.rand(*Z.shape) * 0.1
        row_cluseters = np.array([0, 0, 1, 1])
        col_clusters = np.array([0, 0, 0, 1, 1])
        km = Kmeans(Z=Z,
                    clusters=[row_cluseters, col_clusters],
                    nclusters=[2, 2],
                    k_range=range(2, 4))
        res1 = km.compute()
        self.assertEqual(res1.k_value, 2)
        km = Kmeans(Z=Z,
                    clusters=[row_cluseters, col_clusters],
                    nclusters=[2, 2],
                    k_range=range(3, 1, -1))
        res2 = km.compute()
        self.assertEqual(res2.k_value, 2)
