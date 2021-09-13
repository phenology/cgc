import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from .results import Results

logger = logging.getLogger(__name__)


class KmeansResults(Results):
    """
    Contains results and metadata of a k-means refinement calculation
    """
    def reset(self):
        self.k_value = None
        self.measure_list = None
        self.cl_mean_centroids = None


class Kmeans(object):
    def __init__(self,
                 Z,
                 row_clusters,
                 col_clusters,
                 n_row_clusters,
                 n_col_clusters,
                 k_range=None,
                 max_k_ratio=0.8,
                 kmean_max_iter=100,
                 output_filename=''):
        """
        Set up Kmeans object.

        :param Z: m x n matrix of spatial-temporal data. Usually each row is a
        time-series of a spatial grid.
        :type Z: class:`numpy.ndarray`
        :param row_clusters: m x 1 row cluster array.
        :type row_clusters: class:`numpy.ndarray`
        :param col_clusters: n x 1 column cluster array.
        :type col_clusters: class:`numpy.ndarray`
        :param n_row_clusters: number of row clusters
        :type n_row_clusters: int
        :param n_col_clusters: number of column clusters
        :type n_col_clusters: int
        :param k_range: range of the number of clusters, i.e. value "k"
        :type k_range: range
        :param max_k_ratio: ratio of the maximum k to the total number of 
                            clusters. it will be ignored if "k_range" is given.
                            defaults to 0.8
        :type max_k_ratio: float, optional
        :param kmean_max_iter: maximum number of iterations of the KMeans
        :type kmean_max_iter: int
        :param output_filename: name of the file where to write the results
        :type output_filename: str
        """

        # Input parameters -----------------
        self.row_clusters = row_clusters
        self.col_clusters = col_clusters
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.kmean_max_iter = kmean_max_iter
        self.output_filename = output_filename

        nonempty_row_cl = len(np.unique(self.row_clusters))
        nonempty_col_cl = len(np.unique(self.col_clusters))
        max_k = nonempty_row_cl * nonempty_col_cl
        if k_range is None:
            self.k_range = list(range(2, int(max_k * max_k_ratio)))
        else:
            self.k_range = list(k_range)
        # Input parameters end -------------

        # Store input parameters in results object
        self.results = KmeansResults(**self.__dict__)

        self.Z = Z

        # Number of row/col clusters should be smaller than the max ID
        # Since ID starts from 0
        if not max(self.row_clusters) < self.n_row_clusters:
            raise ValueError("row_clusters include labels >= n_row_clusters")
        if not max(self.col_clusters) < self.n_col_clusters:
            raise ValueError("col_clusters include labels >= n_col_clusters")

        # Check minimum k
        if not min(self.k_range) >= 2:
            raise ValueError("All k-values in k_range must be >= 2")

        # Check maximum k
        max_k_input = max(self.k_range)
        if max_k_input > max_k:
            raise ValueError("The maximum k-value exceeds the "
                             "number of (non-empty) co-clusters")
        elif max_k_input > max_k * 0.8:
            logger.warning("k_range includes large k-values (80% "
                           "of the number of co-clusters or more)")

    def compute(self):
        """
        Compute statistics for each clustering group.
        Then Loop through the range of k values,
        and compute the averaged Silhouette measure of each k.
        Finally select the k with the maximum Silhouette measure

        :return: k-means result object
        """
        # Get statistic measures
        self._compute_statistic_measures()

        # Search for value k
        silhouette_avg_list = np.array(
            [])  # List of average silhouette measure of each k value
        kmeans_cc_list = []
        for k in self.k_range:
            # Compute Kmean
            kmeans_cc = KMeans(n_clusters=k, max_iter=self.kmean_max_iter).fit(
                self.stat_measures_norm)
            silhouette_avg = silhouette_score(self.stat_measures_norm,
                                              kmeans_cc.labels_)
            silhouette_avg_list = np.append(silhouette_avg_list,
                                            silhouette_avg)
            kmeans_cc_list.append(kmeans_cc)
        idx_k = np.argmax(silhouette_avg_list)
        self.results.measure_list = silhouette_avg_list
        self.results.k_value = self.k_range[idx_k]
        self.kmeans_cc = kmeans_cc_list[idx_k]
        del kmeans_cc_list

        # Scale back centroids of the "mean" dimension
        centroids_norm = self.kmeans_cc.cluster_centers_[:, 0]
        stat_max = np.nanmax(self.stat_measures[:, 0])
        stat_min = np.nanmin(self.stat_measures[:, 0])
        mean_centroids = centroids_norm * (stat_max - stat_min) + stat_min

        # Assign centroids to each cluster cell
        cl_mean_centroids = mean_centroids[self.kmeans_cc.labels_]

        # Reshape the centroids of means to the shape of cluster matrix,
        # taking into account non-constructive row/col cluster
        self.results.cl_mean_centroids = np.full(
            (self.n_row_clusters, self.n_col_clusters), np.nan)
        idx = 0
        for r in np.unique(self.row_clusters):
            for c in np.unique(self.col_clusters):
                self.results.cl_mean_centroids[r, c] = cl_mean_centroids[idx]
                idx = idx + 1

        self.results.write(filename=self.output_filename)
        return self.results

    def _compute_statistic_measures(self):
        """
        Compute 6 statistics: Mean, STD, 5 percentile, 95 percentile, maximum
        and minimum values, for each co-cluster group.
        Normalize them to [0, 1]
        """
        row_clusters = np.unique(self.row_clusters)
        col_clusters = np.unique(self.col_clusters)
        self.stat_measures = np.zeros(
            (len(row_clusters) * len(col_clusters), 6))

        # Loop over co-clusters
        for ir, r in enumerate(row_clusters):
            idx_rows, = np.where(self.row_clusters == r)
            for ic, c in enumerate(col_clusters):
                idx_cols, = np.where(self.col_clusters == c)
                rr, cc = np.meshgrid(idx_rows, idx_cols)
                Z = self.Z[rr, cc]

                idx = np.ravel_multi_index(
                    (ir, ic), (len(row_clusters), len(col_clusters)))

                self.stat_measures[idx, 0] = Z.mean()
                self.stat_measures[idx, 1] = Z.std()
                self.stat_measures[idx, 2] = np.percentile(Z, 5)
                self.stat_measures[idx, 3] = np.percentile(Z, 95)
                self.stat_measures[idx, 4] = Z.max()
                self.stat_measures[idx, 5] = Z.min()

        # Normalize all statistics to [0, 1]
        minimum = self.stat_measures.min(axis=0)
        maximum = self.stat_measures.max(axis=0)
        self.stat_measures_norm = np.divide((self.stat_measures - minimum),
                                            (maximum - minimum))

        # Set statistics to zero if all its values are identical (max == min)
        self.stat_measures_norm[np.isnan(self.stat_measures_norm)] = 0.
