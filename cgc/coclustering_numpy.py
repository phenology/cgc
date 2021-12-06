import logging

import numpy as np
import numba

logger = logging.getLogger(__name__)


def _distance(Z, Y):
    """ Distance function """
    # The first term below is equal to one row of: da.dot(da.ones(m, n), Y)
    # with Z.shape = (m, n) and Y.shape = (n, k)
    return Y.sum(axis=0, keepdims=True) - np.dot(Z, np.log(Y))


def _min_dist(Z, M, CoCavg):
    Y = np.dot(M, CoCavg)
    d = _distance(Z, Y)
    return np.argmin(d, axis=1), np.min(d, axis=1)


@numba.jit(nopython=True, nogil=True, cache=True)
def _min_dist_numba(Z, clusters, labels, CoCavg, max=np.nan_to_num(np.inf)):
    m, n = Z.shape
    l, k = CoCavg.shape

    # Extend the matrix of cluster averages to include empty clusters
    CoCavg_ext = np.full((np.max(labels) + 1, k), np.nan)
    CoCavg_ext[labels, :] = CoCavg
    Y = CoCavg_ext[clusters]
    sum = Y.sum(axis=0)

    min_d = np.full(m, max)  # Initialize with largest float
    clusters_new = np.zeros(m, dtype=numba.types.int64)
    for ir in range(k):
        # Calculate distance for cluster ir
        d = sum[ir] - np.dot(Z, np.log(Y[:, ir]))
        # If distance is smaller then previous assignment, reassign
        smaller = d < min_d
        min_d = smaller * d + ~smaller * min_d
        clusters_new = smaller * ir + ~smaller * clusters_new
    return clusters_new, min_d


def _initialize_clusters(n_el, n_clusters):
    """ Initialize cluster array """
    cluster_idx = np.mod(np.arange(n_el), n_clusters)
    return np.random.permutation(cluster_idx)


def _setup_cluster_matrix(cluster_labels, cluster_idx):
    """ Set cluster occupation matrix """
    return np.equal.outer(cluster_idx, cluster_labels)


@numba.jit(nopython=True, nogil=True, parallel=True, cache=True)
def _cluster_dot_numba(Z, row_clusters, col_clusters, row_cluster_labels,
                       col_cluster_labels):
    """
    To replace np.dot(np.dot(R.T, Z), C), where R and C are occupation matrices
    """
    product = np.zeros((len(row_cluster_labels), len(col_cluster_labels)))
    for ircl in range(len(row_cluster_labels)):
        idx_rcl, = np.where(row_clusters == row_cluster_labels[ircl])
        for iccl in range(len(col_cluster_labels)):
            idx_ccl, = np.where(col_clusters == col_cluster_labels[iccl])

            prod_rc = 0
            for idr in idx_rcl:
                for idc in idx_ccl:
                    prod_rc += Z[idr, idc]

            product[ircl, iccl] = prod_rc

    return product


def coclustering(Z,
                 nclusters_row,
                 nclusters_col,
                 errobj,
                 niters,
                 low_memory=False,
                 row_clusters_init=None,
                 col_clusters_init=None):
    """
    Run the co-clustering analysis, Numpy-based implementation.

    :param Z: Data matrix for which to run the co-clustering analysis
    :type Z: numpy.ndarray
    :param nclusters_row: Number of row clusters.
    :type nclusters_row: int
    :param nclusters_col: Number of column clusters.
    :type nclusters_col: int
    :param errobj: Convergence threshold for the objective function.
    :type errobj: float, optional
    :param niters: Maximum number of iterations.
    :type niters: int, optional
    :param low_memory: Make use of a low-memory version of the algorithm with
        Numba JIT acceleration
    :type low_memory: bool, optional
    :param row_clusters_init: Initial row cluster assignment.
    :type row_clusters_init: numpy.ndarray or array_like, optional
    :param col_clusters_init: Initial column cluster assignment.
    :type col_clusters_init: numpy.ndarray or array_like, optional
    :return: Has converged, number of iterations performed, final row and
    column clustering, approximation error of the co-clustering.
    :type: tuple
    """
    [m, n] = Z.shape

    if row_clusters_init is not None:
        row_clusters = np.array(row_clusters_init)
    else:
        row_clusters = _initialize_clusters(m, nclusters_row)

    if col_clusters_init is not None:
        col_clusters = np.array(col_clusters_init)
    else:
        col_clusters = _initialize_clusters(n, nclusters_col)

    e, old_e = 2 * errobj, 0
    s = 0
    converged = False

    while (not converged) & (s < niters):
        logger.debug(f'Iteration # {s} ..')
        # Calculate cluster based averages
        nel_row_clusters = np.bincount(row_clusters, minlength=nclusters_row)
        nel_col_clusters = np.bincount(col_clusters, minlength=nclusters_col)
        row_cluster_labels, = nel_row_clusters.nonzero()
        col_cluster_labels, = nel_col_clusters.nonzero()
        logger.debug('num of populated clusters: row {}, col {}'.format(
            len(row_cluster_labels), len(col_cluster_labels)))
        nel_clusters = np.outer(nel_row_clusters[row_cluster_labels],
                                nel_col_clusters[col_cluster_labels])
        if low_memory:
            CoCavg = _cluster_dot_numba(Z, row_clusters, col_clusters,
                                        row_cluster_labels, col_cluster_labels)
        else:
            R = _setup_cluster_matrix(row_cluster_labels, row_clusters)
            C = _setup_cluster_matrix(col_cluster_labels, col_clusters)
            CoCavg = np.dot(np.dot(R.T, Z), C)
        CoCavg = CoCavg / nel_clusters

        # Calculate distances based on approximation and assign best clusters
        if low_memory:
            _row_clusters, _ = _min_dist_numba(Z, col_clusters,
                                               col_cluster_labels, CoCavg.T)
            col_clusters, dist = _min_dist_numba(Z.T, row_clusters,
                                                 row_cluster_labels, CoCavg)
            row_clusters = _row_clusters
        else:
            row_clusters, _ = _min_dist(Z, C, CoCavg.T)
            col_clusters, dist = _min_dist(Z.T, R, CoCavg)

        row_clusters = np.take(row_cluster_labels, row_clusters)
        col_clusters = np.take(col_cluster_labels, col_clusters)

        # Error value (actually just the column components really)
        old_e = e
        # power 1 divergence, power 2 euclidean
        e = np.sum(np.power(dist, 1))

        logger.debug(f'Error = {e:+.15e}, dE = {e - old_e:+.15e}')
        converged = abs(e - old_e) < errobj
        s = s + 1
    if converged:
        logger.debug(f'Coclustering converged in {s} iterations')
    else:
        logger.debug(f'Coclustering not converged in {s} iterations')
    return converged, s, row_clusters, col_clusters, e
