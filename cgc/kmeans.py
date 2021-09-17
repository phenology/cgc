import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .results import Results
from .utils import calculate_cluster_feature

logger = logging.getLogger(__name__)


class KmeansResults(Results):
    """
    Contains results and metadata of a k-means refinement calculation.

    :var k_value: Optimal K value (value with maximum Silhouette score).
    :type k_value: int
    :var measure_list: List of Silhouette coefficients for all tested k values.
    :type measure_list: np.ndarray
    :var cl_mean_centroids: Refined cluster averages computed as the centroids
        of the clusters resulting from the k-means analysis using `k=k_value`.
        Initally empty clusters are assigned NaN values.
    :type cl_mean_centroids: np.ndarray
    """
    k_value = None
    measure_list = None
    cl_mean_centroids = None


class Kmeans(object):
    """
    Perform a clustering refinement using k-means.

    K-means clustering is performed for multiple k values, then the optimal
    value is selected on the basis of the Silhouette coefficient.

    :param Z: Data array (N dimensions).
    :type Z: numpy.ndarray or dask.array.Array
    :param clusters: Iterable with length N. It should contain the cluster
        labels for each dimension, following the same ordering as for Z.
    :type clusters: tuple, list, or numpy.ndarray
    :param nclusters: Iterable with length N. It should contains the number of
        clusters in each dimension, following the same ordering as for Z.
    :type nclusters: tuple, list, or numpy.ndarray
    :param k_range: Range of k values to test. Default from 2 to
        a fraction of the number of non-empty clusters (see max_k_ratio).
    :type k_range: tuple, list, or numpy.ndarray, optional
    :param max_k_ratio: If k_range is not provided, test all k values from 2
        to `max_k_ratio*max_k`, where `max_k` is the number of non-empty co- or
        tri-clusters. It will be ignored if `k_range` is given. Default to 0.8.
    :type max_k_ratio: float, optional
    :param kmean_max_iter: Maximum number of iterations of k-means.
    :type kmean_max_iter: int, optional
    :param output_filename: Name of the file where to write the results.
    :type output_filename: str, optional
    """
    def __init__(self,
                 Z,
                 clusters,
                 nclusters,
                 k_range=None,
                 max_k_ratio=0.8,
                 kmean_max_iter=100,
                 output_filename=''):
        # Input parameters -----------------
        self.clusters = clusters
        self.nclusters = nclusters
        self.kmean_max_iter = kmean_max_iter
        self.output_filename = output_filename

        max_k = np.prod(self.nclusters)
        if k_range is None:
            self.k_range = list(range(2, int(max_k * max_k_ratio)))
        else:
            self.k_range = list(k_range)
        self.k_range.sort()
        # Input parameters end -------------

        # Store input parameters in results object
        self.results = KmeansResults(**self.__dict__)

        self.Z = Z

        # Check if Z matches the clusters
        if Z.ndim != len(clusters):
            raise ValueError("The number of dimensions of Z is not equal to "
                             "the number of labels provided: "
                             "{} != {}".format(Z.ndim, len(clusters)))
        if Z.shape != tuple(len(cl) for cl in clusters):
            raise ValueError("The shape of Z does not match the shape of the "
                             "clusters: {} != {}".format(
                                 Z.shape, tuple(len(cl) for cl in clusters)))

        # The max label per cluster should be smaller than the number of
        # clusters. Label starts from 0.
        for cl, ncl, id in zip(clusters, nclusters, range(Z.ndim)):
            if not max(cl) < ncl:
                raise ValueError(
                    "One label array includes elements >= number of clusters. "
                    "Cluster dimension order: {}. Label {} >=  ncluster {}.".
                    format(id, max(cl), ncl))

        # Check minimum k
        if not min(self.k_range) >= 2:
            raise ValueError("All k-values in k_range must be >= 2")

        # Check maximum k
        max_k_input = max(self.k_range)
        if max_k_input > max_k:
            raise ValueError("The maximum k-value exceeds the "
                             "number of (non-empty) clusters")
        elif max_k_input > max_k * 0.8:
            logger.warning("k_range includes large k-values (80% "
                           "of the number of clusters or more)")

    def compute(self):
        """
        Compute statistics for each clustering group. Then loop through the
        range of k values, and compute the averaged Silhouette measure of each
        k value. Finally select the k with the maximum Silhouette measure.

        :return: K-means results.
        :type: cgc.kmeans.KmeansResults
        """
        # Get statistic measures
        self._compute_statistic_measures()

        # Search for value k
        silhouette_avg_list = np.array(
            [])  # List of average silhouette measure of each k value
        kmean_cluster_list = []
        for k in self.k_range:
            # Compute Kmean
            kmean_cluster = KMeans(n_clusters=k,
                                   max_iter=self.kmean_max_iter).fit(
                                       self.stat_measures_norm)
            silhouette_avg = silhouette_score(self.stat_measures_norm,
                                              kmean_cluster.labels_)
            silhouette_avg_list = np.append(silhouette_avg_list,
                                            silhouette_avg)
            kmean_cluster_list.append(kmean_cluster)
        idx_k = np.argmax(silhouette_avg_list)
        if np.sum(silhouette_avg_list == silhouette_avg_list[idx_k]) > 1:
            idx_k_list = np.argwhere(
                silhouette_avg_list == silhouette_avg_list[idx_k]).reshape(
                    -1).tolist()
            logger.warning(
                "Multiple k values with the same silhouette score: {},"
                "picking the smallest one: {}".
                format([self.k_range[i] for i in idx_k_list],
                       self.k_range[idx_k]))
        self.results.measure_list = silhouette_avg_list
        self.results.k_value = self.k_range[idx_k]
        self.kmean_cluster = kmean_cluster_list[idx_k]
        del kmean_cluster_list

        # Scale back centroids of the "mean" dimension
        centroids_norm = self.kmean_cluster.cluster_centers_[:, 0]
        stat_max = np.nanmax(self.stat_measures[:, 0])
        stat_min = np.nanmin(self.stat_measures[:, 0])
        mean_centroids = centroids_norm * (stat_max - stat_min) + stat_min

        # Assign centroids to each cluster cell
        cl_mean_centroids = mean_centroids[self.kmean_cluster.labels_]

        # Reshape the centroids of means to the shape of cluster matrix,
        # taking into account non-constructive clusters
        self.results.cl_mean_centroids = np.full(self.nclusters, np.nan)
        idx = 0
        cls_exist = np.array(
            np.meshgrid(*[np.unique(cl) for cl in self.clusters])).T.reshape(
                -1, len(self.nclusters))
        for cl in cls_exist:
            self.results.cl_mean_centroids[tuple(cl)] = cl_mean_centroids[idx]
            idx = idx + 1

        self.results.write(filename=self.output_filename)
        return self.results

    def _compute_statistic_measures(self):
        """
        Compute 6 statistics: Mean, STD, 5 percentile, 95 percentile, maximum
        and minimum values, for each cluster group.
        Normalize them to [0, 1].
        """

        features = np.zeros((*self.nclusters, 6))
        features[...,
                 0] = calculate_cluster_feature(self.Z, np.mean, self.clusters,
                                                self.nclusters)
        features[...,
                 1] = calculate_cluster_feature(self.Z, np.std, self.clusters,
                                                self.nclusters)
        features[..., 2] = calculate_cluster_feature(self.Z,
                                                     np.percentile,
                                                     self.clusters,
                                                     self.nclusters,
                                                     q=5)
        features[..., 3] = calculate_cluster_feature(self.Z,
                                                     np.percentile,
                                                     self.clusters,
                                                     self.nclusters,
                                                     q=95)
        features[...,
                 4] = calculate_cluster_feature(self.Z, np.max, self.clusters,
                                                self.nclusters)
        features[...,
                 5] = calculate_cluster_feature(self.Z, np.min, self.clusters,
                                                self.nclusters)

        self.stat_measures = features[~np.isnan(features)].reshape((-1, 6))

        # Normalize all statistics to [0, 1]
        minimum = self.stat_measures.min(axis=0)
        maximum = self.stat_measures.max(axis=0)
        self.stat_measures_norm = np.divide((self.stat_measures - minimum),
                                            (maximum - minimum))

        # Set statistics to zero if all its values are identical (max == min)
        self.stat_measures_norm[np.isnan(self.stat_measures_norm)] = 0.
