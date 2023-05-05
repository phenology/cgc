import unittest
import numpy as np

from cgc.kmeans import KMeans


def init_cocluster_km():
    """
    Z:
        [[0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 0, 0]]

    cluster index:
        [[(0,0), (0,0), (0,1), (0,1)],
        [(0,0), (0,0), (0,1), (0,1)],
        [(1,0), (1,0), (1,1), (1,1)],
        [(1,0), (1,0), (1,1), (1,1)],
        [(1,0), (1,0), (1,1), (1,1)]]
    """
    Z = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0],
                  [1, 1, 0, 0]])
    row_clusters = np.array([0, 0, 1, 1, 1])
    col_clusters = np.array([0, 0, 1, 1])
    nrow_clusters, ncol_clusters = 3, 2  # 1 non populated row/col cluster
    clusters = [row_clusters, col_clusters]
    nclusters = [nrow_clusters, ncol_clusters]
    k_range = range(2, 4)
    kmeans_max_iter = 100
    km = KMeans(Z=Z,
                clusters=clusters,
                nclusters=nclusters,
                k_range=k_range,
                kmeans_max_iter=kmeans_max_iter)
    return km


def init_cocluster_km_with_noise():
    Z = np.array([
        [0.0, 0.2, 1.0, 1.1],
        [0.2, 0.0, 1.1, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [1.2, 1.2, 0.1, 0.1],
        [1.0, 1.2, 0.0, 0.1]
    ])
    row_clusters = np.array([0, 0, 1, 1, 1])
    col_clusters = np.array([0, 0, 1, 1])
    nrow_clusters, ncol_clusters = 3, 2  # 1 non populated row/col cluster
    clusters = [row_clusters, col_clusters]
    nclusters = [nrow_clusters, ncol_clusters]
    k_range = range(2, 4)
    kmeans_max_iter = 2
    km = KMeans(Z=Z,
                clusters=clusters,
                nclusters=nclusters,
                k_range=k_range,
                kmeans_max_iter=kmeans_max_iter)
    return km


def init_tricluster_km():
    Z1 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]])
    Z = np.full((4, 3, 4), 0)
    Z[:2] = Z1
    Z[2:] = Z1 + 1
    row_clusters = np.array([0, 0, 1])
    col_clusters = np.array([0, 0, 1, 1])
    band_clusters = np.array([0, 0, 1, 1])
    nrow_clusters, ncol_clusters, nband_clusters = 2, 2, 2
    clusters = [band_clusters, row_clusters, col_clusters]
    nclusters = [nband_clusters, nrow_clusters, ncol_clusters]
    k_range = range(2, 4)
    kmeans_max_iter = 2
    km = KMeans(Z=Z,
                clusters=clusters,
                nclusters=nclusters,
                k_range=k_range,
                kmeans_max_iter=kmeans_max_iter)
    return km


class TestKMeans(unittest.TestCase):
    def test_Z_and_cluster_shape_not_match(self):
        with self.assertRaises(ValueError):
            KMeans(Z=np.random.random((5, 5)),
                   clusters=[[0, 0, 1, 1, 2], [0, 0, 1, 1]],
                   nclusters=[3, 2])

    def test_Z_and_cluster_dimension_not_match(self):
        with self.assertRaises(ValueError):
            KMeans(Z=np.random.random((5, 4)),
                   clusters=[[0, 0, 1, 1, 2], [0, 0, 1, 1], [0, 0, 1, 1]],
                   nclusters=[3, 2, 2])

    def test_max_label_equal_ncluster(self):
        with self.assertRaises(ValueError):
            KMeans(Z=np.random.random((5, 4, 4)),
                   clusters=[[0, 0, 1, 1, 2], [0, 0, 1, 1], [0, 0, 1, 1]],
                   nclusters=[3, 2, 1])

    def test_max_label_exceeds_ncluster(self):
        with self.assertRaises(ValueError):
            KMeans(Z=np.random.random((5, 4, 4)),
                   clusters=[[0, 0, 1, 1, 2], [0, 0, 1, 1], [0, 0, 1, 1]],
                   nclusters=[2, 2, 2])

    def test_kvalues_exceed_number_of_coclusters(self):
        with self.assertRaises(ValueError):
            KMeans(
                Z=np.random.random((6, 4)),
                clusters=[[0, 0, 1, 1, 2, 2], [0, 0, 1, 1]],
                nclusters=[3, 2],
                k_range=range(1, 8),
            )

    def test_kvalues_exceed_number_of_coclusters_populated(self):
        with self.assertRaises(ValueError):
            KMeans(
                Z=np.random.random((6, 4)),
                clusters=[[0, 0, 1, 1, 2, 2], [0, 0, 1, 1]],
                nclusters=[4, 2],
                k_range=range(1, 8),
            )

    def test_statistic_coclustering(self):
        km = init_cocluster_km()
        km._compute_statistic_measures()
        results = np.array([[0., 0., 0., 0., 0., 0.], [1., 0., 1., 1., 1., 1.],
                            [1., 0., 1., 1., 1., 1.], [0., 0., 0., 0., 0.,
                                                       0.]])
        self.assertTrue(np.all(results == km.stat_measures_norm))

    def test_smaller_number_of_actual_clusters(self):
        # should not fail even if the number of identified clusters is smaller
        # than the number of required clusters
        km = init_cocluster_km()
        # there are 2 actual clusters
        km.k_range = [3]
        results = km.compute()
        assert results.k_value == 3

    def test_kmeans_labels_coclustering(self):
        km = init_cocluster_km()
        km.compute()

        cl_km_labels = km.results.labels  # Refined labels in clusters
        row_clusters, col_clusters = np.meshgrid(km.clusters[0],
                                                 km.clusters[1],
                                                 indexing='ij')
        Z_km_labels = cl_km_labels[row_clusters,
                                   col_clusters]  # Refined labels in Z

        # Check all values within the same refined cluster are the same
        for k in range(km.results.k_value):
            vk = km.Z[np.where(Z_km_labels == k)]
            self.assertTrue(np.all(vk == vk[0]))

    def test_statistic_triclustering(self):
        km = init_tricluster_km()
        km._compute_statistic_measures()
        results = np.array([[0.,   0., 0., 0., 0., 0.],
                            [0.5, 0., 0.5, 0.5, 0.5, 0.5],
                            [0.5, 0., 0.5, 0.5, 0.5, 0.5],
                            [0., 0., 0., 0., 0., 0.],
                            [0.5, 0., 0.5, 0.5, 0.5, 0.5],
                            [1., 0., 1., 1., 1., 1.],
                            [1., 0., 1., 1., 1., 1.],
                            [0.5, 0., 0.5, 0.5, 0.5, 0.5]])
        self.assertTrue(np.all(results == km.stat_measures_norm))

    def test_kvalues_triclustering(self):
        km = init_tricluster_km()
        km.compute()
        self.assertEqual(km.results.k_value, 3)

    def test_kmeans_labels_triclustering(self):
        km = init_tricluster_km()
        km.compute()

        cl_km_labels = km.results.labels  # Refined labels in clusters
        band_clusters, row_clusters, col_clusters = np.meshgrid(km.clusters[0],
                                                                km.clusters[1],
                                                                km.clusters[2],
                                                                indexing='ij')
        Z_km_labels = cl_km_labels[band_clusters, row_clusters,
                                   col_clusters]  # Refined labels in Z

        # Check all values within the same refined cluster are the same
        for k in range(km.results.k_value):
            vk = km.Z[np.where(Z_km_labels == k)]
            self.assertTrue(np.all(vk == vk[0]))

    def test_centroid_values(self):
        km = init_cocluster_km()
        km.compute()
        self.assertEqual((3, 2), km.results.cluster_averages.shape)
        target_centroids = np.array([
            [0., 1.],
            [1., 0.],
            [np.nan, np.nan]
        ])
        self.assertTrue(
            np.allclose(
                target_centroids,
                km.results.cluster_averages,
                equal_nan=True
            )
        )

    def test_centroid_values_for_km_with_noise(self):
        km = init_cocluster_km_with_noise()
        km.compute()
        self.assertEqual((3, 2), km.results.cluster_averages.shape)
        target_centroids = np.array([
            [0.07, 1.08],
            [1.08, 0.07],
            [np.nan, np.nan]
        ])
        self.assertTrue(
            np.allclose(
                target_centroids,
                km.results.cluster_averages,
                equal_nan=True
            )
        )

    def test_centroids_nan(self):
        km = init_cocluster_km()
        km.compute()
        self.assertTrue(all(np.isnan(km.results.cluster_averages[2, :])))

    def test_kvalue_does_not_depend_on_krange_order(self):
        # 4 co-clusters, 2 clusters
        Z = np.array([[1, 1, 1, 2, 2], [1, 1, 1, 2, 2], [2, 2, 2, 1, 1],
                      [2, 2, 2, 1, 1]])
        Z = Z + np.random.rand(*Z.shape) * 0.1
        row_cluseters = np.array([0, 0, 1, 1])
        col_clusters = np.array([0, 0, 0, 1, 1])
        km = KMeans(Z=Z,
                    clusters=[row_cluseters, col_clusters],
                    nclusters=[2, 2],
                    k_range=range(2, 4))
        res1 = km.compute()
        self.assertEqual(res1.k_value, 2)
        km = KMeans(Z=Z,
                    clusters=[row_cluseters, col_clusters],
                    nclusters=[2, 2],
                    k_range=range(3, 1, -1))
        res2 = km.compute()
        self.assertEqual(res2.k_value, 2)

    def test_user_defined_statistics(self):
        # use optional parameters for one of the statistics function
        Z = np.array([[0, 0, 1, 1],
                      [0, 0, 1, 1],
                      [1, 1, 2, 2],
                      [1, 1, 2, 2],
                      [1, 1, 2, 2]])
        row_clusters = np.array([0, 0, 1, 1, 1])
        col_clusters = np.array([0, 0, 1, 1])
        nrow_clusters, ncol_clusters = 2, 2
        clusters = [row_clusters, col_clusters]
        nclusters = [nrow_clusters, ncol_clusters]
        k_range = [2, 3]
        kmeans_max_iter = 100
        km = KMeans(Z=Z,
                    clusters=clusters,
                    nclusters=nclusters,
                    k_range=k_range,
                    kmeans_max_iter=kmeans_max_iter,
                    statistics=(np.mean, (np.std, {'axis': None})))
        res = km.compute()
        assert res.input_parameters["statistics"][0][0] == "mean"
        assert len(res.input_parameters["statistics"][0][1]) == 0
        assert res.input_parameters["statistics"][1][0] == "std"
        assert len(res.input_parameters["statistics"][1][1]) == 1
        assert res.k_value == 3

    def test_check_values_with_user_defined_statistics(self):
        # the choice of the statistics affects the results
        Z = np.array([[0, 0, 1, 1],
                      [0, 0, 0.5, 1],
                      [1, 1, 2, 2],
                      [1, 1, 2, 2],
                      [1, 0.5, 2, 0.5]])
        row_clusters = np.array([0, 0, 1, 1, 1])
        col_clusters = np.array([0, 0, 1, 1])
        nrow_clusters, ncol_clusters = 2, 2
        clusters = [row_clusters, col_clusters]
        nclusters = [nrow_clusters, ncol_clusters]
        k_range = [2, 3]
        kmeans_max_iter = 100

        def run_kmeans(statistics=None):
            km = KMeans(Z=Z,
                        clusters=clusters,
                        nclusters=nclusters,
                        k_range=k_range,
                        kmeans_max_iter=kmeans_max_iter,
                        statistics=statistics)
            res = km.compute()
            return res.k_value

        assert run_kmeans() == 3
        assert run_kmeans((np.min,)) == 2
