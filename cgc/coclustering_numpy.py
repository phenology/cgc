import logging

import numpy as np
import numba

logger = logging.getLogger(__name__)


def _distance(Z, Y, epsilon):
    """ Distance function """
    Y = Y + epsilon
    # The first term below is equal to one row of: da.dot(da.ones(m, n), Y)
    # with Z.shape = (m, n) and Y.shape = (n, k)
    return Y.sum(axis=0, keepdims=True) - np.dot(Z, np.log(Y))


def _min_dist(Z, M, CoCavg, epsilon):
    Y = np.dot(M, CoCavg)
    d = _distance(Z, Y, epsilon)
    return np.argmin(d, axis=1), np.min(d, axis=1)


def _min_dist_lowmem(Z, clusters, CoCavg, epsilon):
    m, n = Z.shape
    l, k = CoCavg.shape
    Y = CoCavg[clusters] + epsilon
    sum = Y.sum(axis=0)
    min_d = np.full(m, np.nan_to_num(np.inf))  # Initialize with largest float
    clusters_new = np.zeros(m, dtype=np.int)
    for ir in range(k):
        # Calculate distance for cluster ir
        d = sum[ir] - np.dot(Z, np.log(Y[:, ir]))
        # If distance is smaller then previous assignment, reassign
        smaller = d < min_d
        min_d = smaller * d + ~smaller * min_d
        clusters_new = smaller * ir + ~smaller * clusters_new
    return clusters_new, min_d


@numba.jit(nopython=True, nogil=True, cache=True)
def _min_dist_numba(Z, clusters, CoCavg, epsilon, max=np.nan_to_num(np.inf)):
    m, n = Z.shape
    l, k = CoCavg.shape
    Y = CoCavg[clusters] + epsilon
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


def _setup_cluster_matrix(n_clusters, cluster_idx):
    """ Set cluster occupation matrix """
    return np.eye(n_clusters, dtype=np.int32)[cluster_idx]


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


def coclustering(Z,
                 nclusters_row,
                 nclusters_col,
                 errobj,
                 niters,
                 epsilon,
                 low_memory=False,
                 numba_jit=False,
                 row_clusters_init=None,
                 col_clusters_init=None):
    """
    Run the co-clustering, Numpy-based implementation

    :param Z: m x n data matrix
    :param nclusters_row: number of row clusters
    :param nclusters_col: number of column clusters
    :param errobj: convergence threshold for the objective function
    :param niters: maximum number of iterations
    :param epsilon: numerical parameter, avoids zero arguments in log
    :param low_memory: boolean, set low memory usage version
    :param numba_jit: boolean, set numba optimized single node  version
    :param row_clusters_init: initial row cluster assignment
    :param col_clusters_init: initial column cluster assignment
    :return: has converged, number of iterations performed, final row and
    column clustering, error value
    """
    [m, n] = Z.shape

    if row_clusters_init is not None:
        row_clusters = np.array(row_clusters_init)
    else:
        row_clusters = _initialize_clusters(m, nclusters_row)
    if not low_memory:
        R = _setup_cluster_matrix(nclusters_row, row_clusters)

    if col_clusters_init is not None:
        col_clusters = np.array(col_clusters_init)
    else:
        col_clusters = _initialize_clusters(n, nclusters_col)
    if not low_memory:
        C = _setup_cluster_matrix(nclusters_col, col_clusters)

    e, old_e = 2 * errobj, 0
    s = 0
    converged = False

    Gavg = Z.mean()

    while (not converged) & (s < niters):
        logger.debug(f'Iteration # {s} ..')
        # Calculate cluster based averages
        nel_row_clusters = np.bincount(row_clusters, minlength=nclusters_row)
        nel_col_clusters = np.bincount(col_clusters, minlength=nclusters_col)
        logger.debug('num of populated clusters: row {}, col {}'.format(
            np.sum(nel_row_clusters > 0), np.sum(nel_col_clusters > 0)))
        nel_clusters = np.outer(nel_row_clusters, nel_col_clusters)
        if low_memory:
            if numba_jit:
                CoCavg = _cluster_dot_numba(Z, row_clusters, col_clusters,
                                            nclusters_row, nclusters_col)
            else:
                CoCavg = _cluster_dot(Z, row_clusters, col_clusters,
                                      nclusters_row, nclusters_col)
        else:
            CoCavg = np.dot(np.dot(R.T, Z), C)
        CoCavg += Gavg * epsilon
        CoCavg /= nel_clusters + epsilon

        # Calculate distance based on row approximation and assign best cluster
        if low_memory:
            if numba_jit:
                row_clusters, _ = _min_dist_numba(Z, col_clusters, CoCavg.T,
                                                  epsilon)
            else:
                row_clusters, _ = _min_dist_lowmem(Z, col_clusters, CoCavg.T,
                                                   epsilon)
        else:
            row_clusters, _ = _min_dist(Z, C, CoCavg.T, epsilon)
            R = _setup_cluster_matrix(nclusters_row, row_clusters)

        # Calculate distance based on col approximation and assign best cluster
        if low_memory:
            if numba_jit:
                col_clusters, dist = _min_dist_numba(Z.T, row_clusters, CoCavg,
                                                     epsilon)
            else:
                col_clusters, dist = _min_dist_lowmem(Z.T, row_clusters,
                                                      CoCavg, epsilon)
        else:
            col_clusters, dist = _min_dist(Z.T, R, CoCavg, epsilon)
            C = _setup_cluster_matrix(nclusters_col, col_clusters)

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
