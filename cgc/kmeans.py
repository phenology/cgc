import copy
import numpy as np
import logging
import sklearn.cluster

from sklearn.metrics import silhouette_score

from .results import Results
from .utils import calculate_cluster_feature

logger = logging.getLogger(__name__)

DEFAULT_STATISTICS = [
    (np.mean, None),
    (np.std, None),
    (np.percentile, {"q": 5}),
    (np.percentile, {"q": 95}),
    (np.max, None),
    (np.min, None),
]


class KMeansResults(Results):
    """
    Contains results and metadata of a k-means refinement calculation.

    :var k_value: Optimal K value (value with maximum Silhouette score).
    :type k_value: int
    :var labels: Refined clusters labels. It is a 2D- (for coclustering)
                 or 3D- (for triclustering) array, with the shape of
                 `nclusters`. The value at location (band, row, column)
                 represents the refined cluster label of the corresponding
                 band/row/column cluster combination.
    :type labels: np.ndarray
    :var measure_list: List of Silhouette coefficients for all tested k values.
    :type measure_list: np.ndarray
    :var cluster_averages: Refined cluster averages. They are computed as means
        over all elements of the co-/tri-clusters assigned to the refined
        clusters. Initially empty clusters are assigned NaN values.
    :type cluster_averages: np.ndarray
    """
    k_value = None
    labels = None
    measure_list = None
    cluster_averages = None


class KMeans(object):
    """
    Perform a clustering refinement using k-means.

    A set of statistics is computed for all co- or tri-clusters, then these
    clusters are in turned grouped using k-means. K-means clustering is
    performed for multiple k values, then the optimal value is selected on the
    basis of the Silhouette coefficient.

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
    :param kmeans_max_iter: Maximum number of iterations of k-means.
    :type kmeans_max_iter: int, optional
    :param statistics: Statistics to be computed over the clusters, which are
        then used to refine these. These are provided as an iterable of
        callable functions, with optional keyword arguments. For example:
        [(func1, {'kwarg1': val1, ...}), (func2, {'kwarg2': val2, ...}, ...] .
        See cgc.kmeans.DEFAULT_STATISTICS for the default statistics, and
        cgc.utils.calculate_cluster_feature for input function requirements.
    :type statistics: tuple or list, optional
    :param output_filename: Name of the file where to write the results.
    :type output_filename: str, optional

    :Example:

    >>> import numpy as np
    >>> Z = np.array([[4, 4, 1, 1], [4, 4, 1, 1], [2, 2, 3, 3], [2, 2, 3, 3],
                   [2, 2, 3, 3]])
    >>> clusters = [np.array([0, 0, 1, 1, 1]), np.array([0, 0, 1, 1])]
    >>> km = KMeans(Z=Z,
                clusters=clusters,
                nclusters=[2, 2],
                k_range= range(2, 4),
                kmeans_max_iter=2)
    """
    def __init__(self,
                 Z,
                 clusters,
                 nclusters,
                 k_range=None,
                 max_k_ratio=0.8,
                 kmeans_max_iter=100,
                 statistics=None,
                 output_filename=''):
        # Input parameters -----------------
        self.clusters = clusters
        self.nclusters = nclusters
        self.kmeans_max_iter = kmeans_max_iter
        self.output_filename = output_filename

        max_k = np.prod(self.nclusters)
        if k_range is None:
            self.k_range = list(range(2, int(max_k * max_k_ratio)))
        else:
            self.k_range = list(k_range)
        self.k_range.sort()

        statistics = DEFAULT_STATISTICS if statistics is None else statistics
        self.statistics = []
        for el in statistics:
            if hasattr(el, "__call__"):
                func, kwargs = el, dict()
            else:
                func, kwargs = el
                kwargs = kwargs if kwargs is not None else dict()
            self.statistics.append((func, kwargs))
        # Input parameters end -------------

        # Store input parameters in results object
        self.results = KMeansResults(
            clusters=self.clusters,
            nclusters=self.nclusters,
            kmean_max_iter=self.kmeans_max_iter,
            output_filename=self.output_filename,
            k_range=self.k_range,
            statistics=[(func.__name__, kw) for func, kw in self.statistics],
        )

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

        self.stat_measures_norm = None

    def compute(self, recalc_statistics=False):
        """
        Compute statistics for each clustering group. Then loop through the
        range of k values, and compute the averaged Silhouette measure of each
        k value. Finally select the k with the maximum Silhouette measure.

        :param recalc_statistics: If True, always recompute statistics.
        :type recalc_statistics: bool, optional
        :return: K-means results.
        :type: cgc.kmeans.KMeansResults
        """
        # Compute statistics
        if self.stat_measures_norm is None or recalc_statistics:
            self._compute_statistic_measures()

        # Search for value k
        silhouette_avg_list = np.array([])  # average silhouette measure vs k
        kmeans_label_list = []
        for k in self.k_range:
            # Compute Kmean
            km = sklearn.cluster.KMeans(
                n_clusters=k, max_iter=self.kmeans_max_iter
            )
            kmeans_cluster = km.fit(self.stat_measures_norm)
            silhouette_avg = silhouette_score(self.stat_measures_norm,
                                              kmeans_cluster.labels_)
            silhouette_avg_list = np.append(silhouette_avg_list,
                                            silhouette_avg)
            kmeans_label_list.append(kmeans_cluster.labels_)
        idx_k = np.argmax(silhouette_avg_list)
        if np.sum(silhouette_avg_list == silhouette_avg_list[idx_k]) > 1:
            idx_k_list = np.argwhere(
                silhouette_avg_list == silhouette_avg_list[idx_k]).reshape(
                    -1).tolist()
            logger.warning(
                "Multiple k values with the same silhouette score: {},"
                "picking the smallest one: {}".format(
                    [self.k_range[i] for i in idx_k_list],
                    self.k_range[idx_k]))
        self.results.measure_list = silhouette_avg_list
        self.results.k_value = self.k_range[idx_k]
        labels = kmeans_label_list[idx_k]

        indices = np.meshgrid(*[np.unique(cl) for cl in self.clusters],
                              indexing='ij')
        mask = np.zeros(self.nclusters, dtype=bool)
        mask[tuple(indices)] = True

        # Make a lookup matrix from un-refined clusters to Kmean clusters
        km_labels = np.full(self.nclusters, np.nan)
        km_labels[mask] = labels
        self.results.labels = km_labels

        # Calculate the means over the refined clusters
        cluster_averages = np.full(self.nclusters, np.nan)
        for label in range(self.results.k_value):
            label_sum = 0.
            label_n_elements = 0
            # Loop over all co-/tri-clusters in the selected refined cluster
            clusters = np.nonzero(km_labels == label)
            if all(len(c) > 0 for c in clusters):
                for cluster in zip(*clusters):
                    idx = [np.where(self.clusters[i] == cluster[i])[0]
                           for i in range(self.Z.ndim)]
                    label_n_elements += np.prod([len(idx_x) for idx_x in idx])
                    idx = np.meshgrid(*idx, indexing='ij')
                    label_sum += self.Z[tuple(idx)].sum()
                cluster_averages[clusters] = label_sum / label_n_elements
        self.results.cluster_averages = cluster_averages

        self.results.write(filename=self.output_filename)
        return copy.copy(self.results)

    def _compute_statistic_measures(self):
        """
        Compute statistics for each cluster group and normalize them to [0, 1].
        """
        nstats = len(self.statistics)
        features = np.zeros((*self.nclusters, nstats))
        for nstat, (func, kwargs) in enumerate(self.statistics):
            features[..., nstat] = calculate_cluster_feature(self.Z,
                                                             func,
                                                             self.clusters,
                                                             self.nclusters,
                                                             **kwargs)

        stat_measures = features[~np.isnan(features)].reshape((-1, nstats))

        # Normalize all statistics to [0, 1]
        minimum = stat_measures.min(axis=0)
        maximum = stat_measures.max(axis=0)
        self.stat_measures_norm = np.divide((stat_measures - minimum),
                                            (maximum - minimum),
                                            out=np.zeros_like(stat_measures),
                                            where=(maximum - minimum) != 0)
