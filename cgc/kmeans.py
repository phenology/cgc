import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class Kmeans(object):
    def __init__(self,
                 Z,
                 row_clusters,
                 col_clusters,
                 n_row_clusters,
                 n_col_clusters,
                 k_range,
                 kmean_max_iter=100,
                 var_thres=2,
                 ouputdir=None):
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
        :param kmean_max_iter: maximum number of iterations of the KMeans
        :type kmean_max_iter: int
        """
        self.Z = Z
        self.row_clusters = row_clusters
        self.col_clusters = col_clusters
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.k_range = k_range
        self.var_thres = var_thres
        self.kmean_max_iter = kmean_max_iter
        self.ouputdir = ouputdir

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
        # Get statistic measures
        self._compute_statistic_mesures()

        # Search for value k
        var_list = np.array([])  # List of variance of each k value
        kmeans_cc_list = []
        for k in self.k_range:
            # Compute Kmean
            kmeans_cc = KMeans(n_clusters=k, max_iter=self.kmean_max_iter).fit(
                self.stat_measures_norm)
            var_list = np.hstack((var_list, self._compute_sum_var(kmeans_cc)))
            kmeans_cc_list.append(kmeans_cc)
        idx_k = max(np.where(var_list > self.var_thres)[0])
        self.k_value = self.k_range[idx_k]
        self.kmeans_cc = kmeans_cc_list[idx_k]
        del kmeans_cc_list

        # Output elbow curve plot
        if self.ouputdir is not None:
            outplot_path = self.ouputdir + '/kmean_curve.png'
            plt.plot(self.k_range, var_list)  # kmean curve
            plt.plot([min(self.k_range), max(self.k_range)],
                     [self.var_thres, self.var_thres],
                     color='r',
                     linestyle='--')  # Threshold
            plt.plot([self.k_value, self.k_value],
                     [min(var_list), max(var_list)],
                     color='g',
                     linestyle='--')  # Selected k
            xtick_step = int((max(self.k_range) - min(self.k_range)) / 6)
            ytick_step = int((max(var_list) - min(var_list)) / 6)
            plt.xticks(range(min(self.k_range), max(self.k_range), xtick_step))
            plt.xlim(min(self.k_range), max(self.k_range))
            plt.ylim(min(var_list), max(var_list))
            plt.text(max(self.k_range) - 2 * xtick_step,
                     self.var_thres + ytick_step / 4,
                     'threshold={}'.format(self.var_thres),
                     color='r',
                     fontsize=12)
            plt.text(self.k_value + xtick_step / 4,
                     max(var_list) - ytick_step,
                     'k={}'.format(self.k_value),
                     color='g',
                     fontsize=12)
            plt.xlabel('k value', fontsize=20)
            plt.ylabel('Sum of variance', fontsize=20)
            plt.grid(True)
            plt.savefig(outplot_path,
                        format='png',
                        transparent=True,
                        bbox_inches="tight")

        # Scale back centroids of the "mean" dimension
        centroids_norm = self.kmeans_cc.cluster_centers_[:, 0]
        stat_max = np.nanmax(self.stat_measures[:, 0])
        stat_min = np.nanmin(self.stat_measures[:, 0])
        mean_centroids = centroids_norm * (stat_max - stat_min) + stat_min

        # Assign centroids to each cluster cell
        cl_mean_centroids = mean_centroids[self.kmeans_cc.labels_]
        
        # Reshape the centroids of means to the shape of cluster matrix, taking into account
        # non-constructive row/col cluster
        self.cl_mean_centroids = np.empty(
            (self.n_row_clusters, self.n_col_clusters))
        self.cl_mean_centroids[:] = np.nan
        idx = 0
        for r in np.unique(self.row_clusters):
            for c in np.unique(self.col_clusters):
                self.cl_mean_centroids[r, c] = cl_mean_centroids[idx]
                idx = idx + 1
    
    def _compute_statistic_mesures(self):
        """
        Compute 6 statistics: Mean, STD, 5 percentile, 95 percentile, maximum
        and minimum values, for each co-cluster group. 
        Normalize them to [0, 1]
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

        # Normalize all statistic measures to [0, 1]
        self.stat_measures_norm = []
        descale = []
        for sm in self.stat_measures.T:
            minimum = np.nanmin(sm, axis=0)
            maximum = np.nanmax(sm, axis=0)
            sm_norm = np.divide((sm - minimum), (maximum - minimum))
            self.stat_measures_norm.append(sm_norm)

        self.stat_measures_norm = np.array(self.stat_measures_norm).T

    def _compute_sum_var(self, kmeans_cc):
        """
        Compute the sum of squared variance of each Kmean cluster
        """

        # Compute the sum of variance of all points
        var_sum = np.sum((self.stat_measures_norm -
                          kmeans_cc.cluster_centers_[kmeans_cc.labels_])**2)

        return var_sum
