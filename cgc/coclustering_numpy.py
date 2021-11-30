import logging

import numpy as np
import numba

logger = logging.getLogger(__name__)


def _distance(Z, Y):
    """ Distance function """
    # The first term below is equal to one row of: da.dot(da.ones(m, n), Y)
    # with Z.shape = (m, n) and Y.shape = (n, k)
    sum = Y.sum(axis=0, keepdims=True)
    logY = np.zeros_like(Y)
    logY = np.log(Y, out=logY, where=~np.isnan(Y))
    return sum - np.dot(Z, logY)


def _min_dist(Z, clusters, CoCavg):
    Y = CoCavg[clusters]
    d = _distance(Z, Y)
    return np.nanargmin(d, axis=1), np.nanmin(d, axis=1)


def _min_dist_lowmem(Z, clusters, CoCavg):
    m, n = Z.shape
    Y = CoCavg[clusters]
    sum = Y.sum(axis=0)
    min_d = np.full(m, np.nan_to_num(np.inf))  # Initialize with largest float
    clusters_new = np.zeros(m, dtype=np.int)
    empty_cluster_mask = np.isnan(CoCavg).all(axis=0)
    populated_clusters, = np.where(~empty_cluster_mask)
    for ir in populated_clusters:
        # Calculate distance for cluster ir
        d = sum[ir] - np.dot(Z, np.log(Y[:, ir]))
        # If distance is smaller then previous assignment, reassign
        smaller = d < min_d
        min_d = smaller * d + ~smaller * min_d
        clusters_new = smaller * ir + ~smaller * clusters_new
    return clusters_new, min_d


@numba.jit(nopython=True, nogil=True, cache=True)
def _min_dist_numba(Z, clusters, CoCavg, max=np.nan_to_num(np.inf)):
    m, _ = Z.shape
    k, _ = CoCavg.shape
    Y = CoCavg[clusters]
    sum = Y.sum(axis=0)
    min_d = np.full(m, max)  # Initialize with largest float
    clusters_new = np.zeros(m, dtype=numba.types.int64)
    empty_cluster_mask = np.isnan(CoCavg).sum(axis=0) == k
    populated_clusters, = np.where(~empty_cluster_mask)
    for icl in range(len(populated_clusters)):
        ir = populated_clusters[icl]
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


def _setup_cluster_matrix(n_clusters, cluster_idx):
    """ Set cluster occupation matrix """
    return np.eye(n_clusters, dtype=np.bool)[cluster_idx]


def _cluster_dot(Z, row_clusters, col_clusters, nclusters_row, nclusters_col):
    """
    To replace np.dot(np.dot(R.T, Z), C), where R and C are full matrix
    """
    product = np.zeros((nclusters_row, nclusters_col))
    for r in range(nclusters_row):
        idx_r = np.where(row_clusters == r)[0]
        for c in range(nclusters_col):
            idx_c = np.where(col_clusters == c)[0]
            ir, ic = np.meshgrid(idx_r, idx_c)
            product[r, c] = Z[ir, ic].sum()
    return product


@numba.jit(nopython=True, nogil=True, parallel=True, cache=True)
def _cluster_dot_numba(Z, row_clusters, col_clusters, nclusters_row,
                       nclusters_col):
    """
    To replace np.dot(np.dot(R.T, Z), C), where R and C are full matrix
    """
    product = np.zeros((nclusters_row, nclusters_col))
    for r in range(nclusters_row):
        idx_r = np.where(row_clusters == r)[0]
        for c in range(nclusters_col):
            idx_c = np.where(col_clusters == c)[0]

            prod_rc = 0
            for idr in idx_r:
                for idc in idx_c:
                    prod_rc += Z[idr, idc]

            product[r, c] = prod_rc

    return product


def coclustering(Z, nclusters_row, nclusters_col, errobj, niters,
                 low_memory=False, numba_jit=False, row_clusters_init=None,
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
    :param low_memory: Make use of a low-memory version of the algorithm.
    :type low_memory: bool, optional
    :param numba_jit: Make use of Numba JIT acceleration (only if low_memory
        is True).
    :type numba_jit: bool, optional
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
        nel_clusters = np.outer(nel_row_clusters, nel_col_clusters)
        pop_clusters_mask = nel_clusters > 0
        logger.debug('num of populated clusters: row {}, col {}'.format(
            pop_clusters_mask.any(axis=1).sum(),
            pop_clusters_mask.any(axis=0).sum()
        ))

        if low_memory:
            if numba_jit:
                CoCavg = _cluster_dot_numba(Z, row_clusters, col_clusters,
                                            nclusters_row, nclusters_col)
            else:
                CoCavg = _cluster_dot(Z, row_clusters, col_clusters,
                                      nclusters_row, nclusters_col)
        else:
            R = _setup_cluster_matrix(nclusters_row, row_clusters)
            C = _setup_cluster_matrix(nclusters_col, col_clusters)
            CoCavg = np.dot(np.dot(R.T, Z), C)
        CoCavg[~pop_clusters_mask] = np.nan
        np.divide(CoCavg, nel_clusters, out=CoCavg, where=pop_clusters_mask)

        # Calculate distances based on approximation and assign best clusters
        if low_memory:
            if numba_jit:
                _row_clusters, _ = _min_dist_numba(Z, col_clusters, CoCavg.T)
                col_clusters, dist = _min_dist_numba(Z.T, row_clusters, CoCavg)
                row_clusters = _row_clusters
            else:
                _row_clusters, _ = _min_dist_lowmem(Z, col_clusters, CoCavg.T)
                col_clusters, dist = _min_dist_lowmem(Z.T, row_clusters,
                                                      CoCavg)
                row_clusters = _row_clusters
        else:
            _row_clusters, _ = _min_dist(Z, col_clusters, CoCavg.T)
            col_clusters, dist = _min_dist(Z.T, row_clusters, CoCavg)
            row_clusters = _row_clusters

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
