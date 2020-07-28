import numpy as np
import logging
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class Kmeans(object):
    def __init__(self,
                 Z,
                 row_clusters,
                 col_clusters,
                 n_row_clusters,
                 n_col_clusters,
                 kmean_n_clusters,
                 kmean_max_iter=100):
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
        :param kmean_n_clusters: number of clusters to form in KMean, i.e.
        value "k"
        :type kmean_n_clusters: int
        :param kmean_max_iter: maximum number of iterations of the KMeans
        :type kmean_max_iter: int
        """
        self.Z = Z
        self.row_clusters = row_clusters
        self.col_clusters = col_clusters
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.kmean_n_clusters = kmean_n_clusters
        self.kmean_max_iter = kmean_max_iter

        if len(np.unique(row_clusters)) > n_row_clusters:
            print('Setting "n_row_clusters" to {}, \
            accoding to the number of unique elements in "row_clusters".'.
                  format(len(np.unique(row_clusters))))
            self.n_row_clusters = len(np.unique(row_clusters))

        if len(np.unique(col_clusters)) > n_col_clusters:
            print('Setting "col_clusters" to {}, \
            accoding to the number of unique elements in "col_clusters".'.
                  format(len(np.unique(col_clusters))))
            self.n_col_clusters = len(np.unique(col_clusters))

    def compute(self):
        """
        Compute statistics for each clustering group.
        Then compute centroids of the "mean value" dimension.
        """
        self._statistic_mesures()
        self._compute_kmean()

    def _statistic_mesures(self):
        """
        Compute 6 statistics: Mean, STD, 5 percentile, 95 percentile, maximum
        and minimum values, for each co-cluster group.
        """
        self.stat_measures = np.empty([0, 6])
        # Loop per co-cluster cell
        for r in np.unique(self.row_clusters):
            for c in np.unique(self.col_clusters):
                idx_rows = np.where(self.row_clusters == r)[0]
                idx_col = np.where(self.col_clusters == c)[0]
                # All elements in Z falling into this cluster cell
                cl_Z = self.Z[idx_rows, :][:, idx_col]

                cl_stat = np.array([
                    np.nanmean(cl_Z),
                    np.nanstd(cl_Z),
                    np.nanpercentile(cl_Z, 5),
                    np.nanpercentile(cl_Z, 95),
                    np.nanmax(cl_Z),
                    np.nanmin(cl_Z)
                ])

                self.stat_measures = np.vstack((self.stat_measures, cl_stat))

    def _compute_kmean(self):
        """
        Compute kmean centroids.
        """
        # Normalize all statistic measures to [0, 1]
        stat_measures_norm = []
        descale = []
        for sm in self.stat_measures.T:
            minimum = np.nanmin(sm, axis=0)
            maximum = np.nanmax(sm, axis=0)
            sm_norm = np.divide((sm - minimum), (maximum - minimum))
            stat_measures_norm.append(sm_norm)

        self.stat_measures_norm = np.array(stat_measures_norm).T

        # Compute Kmean
        self.kmeans_cc = KMeans(n_clusters=self.kmean_n_clusters,
                                max_iter=self.kmean_max_iter).fit(
                                    self.stat_measures_norm)

        # Get centroids of the "mean value" dimension, and scale back
        # TODO: do we need centroids of other statistic measures?
        mean_centroids_norm = self.kmeans_cc.cluster_centers_[:, 0]
        max_mean = np.nanmax(self.stat_measures[:, 0])
        min_mean = np.nanmin(self.stat_measures[:, 0])
        mean_centroids = mean_centroids_norm * (max_mean - min_mean) + min_mean

        # Assign centroids to each cluster cell
        cl_mean_centroids = mean_centroids[self.kmeans_cc.labels_]

        # Reshape to the shape of cluster matrix, taking into account
        # non-constructive row/col cluster
        self.cl_mean_centroids = np.empty(
            (self.n_row_clusters, self.n_col_clusters))
        self.cl_mean_centroids[:] = np.nan
        idx = 0
        for r in np.unique(self.row_clusters):
            for c in np.unique(self.col_clusters):
                self.cl_mean_centroids[r, c] = cl_mean_centroids[idx]
                idx = idx + 1
