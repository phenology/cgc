import numpy as np
import numba


def _distance(Z, X, Y, epsilon):
    """ Distance function """
    Y = Y + epsilon
    d = np.dot(X, Y) - np.dot(Z, np.log(Y))
    return d


def _distance_lowmem(Z, vec, cc, epsilon):
    """ Distance function low memory"""
    dim1 = vec.size
    dim2 = cc.shape[1]
    product = np.zeros((vec.size, cc.shape[1]))
    for cl in np.unique(vec):
        idx = np.where(vec == cl)[0]
        product[idx, :] = cc[cl, :]

    part1 = np.repeat(np.sum(product, axis=0, keepdims=True, dtype='float64'),
                      Z.shape[0],
                      axis=0)
    part2 = Z.shape[1] * epsilon
    part3 = np.dot(Z, np.log(product + epsilon))
    return part1 + part2 - part3


@numba.jit(nopython=True, nogil=True, parallel=True, cache=True)
def _distance_lowmem_numba(Z, vec, cc, epsilon):
    """ Distance function low memory"""
    dim1 = vec.size
    dim2 = cc.shape[1]
    product = np.zeros((dim1, dim2))
    for cl in np.unique(vec):
        idx = np.where(vec == cl)[0]
        product[idx, :] = cc[cl, :]

    sum_part1 = np.sum(product, axis=0)
    Zdim0 = Z.shape[0]
    part1 = np.zeros((Zdim0, dim2))
    for i in range(Zdim0):
        part1[i, :] = sum_part1

    part2 = Z.shape[1] * epsilon
    part3 = np.dot(Z, np.log(product + epsilon))
    return part1 + part2 - part3


def _initialize_clusters(n_el, n_clusters, low_memory=False):
    """ Initialize cluster occupation matrix """
    cluster_idx = np.mod(np.arange(n_el), n_clusters)
    cluster_idx = np.random.permutation(cluster_idx)
    if low_memory:
        return cluster_idx
    else:
        eye = np.eye(n_clusters, dtype=np.int32)
        return eye[cluster_idx]


def _cluster_dot(Z, row_clusters, col_clusters, nclusters_row, nclusters_col):
    """
    To replace np.dot(np.dot(R.T, Z), C), where R and C are full matrix
    """
    product = np.zeros((nclusters_row, nclusters_col))
    for r in range(0, nclusters_row):
        for c in range(0, nclusters_col):
            idx_r = np.where(row_clusters == r)[0]
            idx_c = np.where(col_clusters == c)[0]
            idx_rc = np.array(np.meshgrid(idx_r, idx_c)).T.reshape(-1, 2)
            product[r, c] = np.sum(Z[idx_rc[:, 0], idx_rc[:, 1]])

    return product


@numba.jit(nopython=True, nogil=True, parallel=True, cache=True)
def _cluster_dot_numba(Z, row_clusters, col_clusters, nclusters_row,
                       nclusters_col):
    """
    To replace np.dot(np.dot(R.T, Z), C), where R and C are full matrix
    """
    product = np.zeros((nclusters_row, nclusters_col))
    for r in range(0, nclusters_row):
        for c in range(0, nclusters_col):
            idx_r = np.where(row_clusters == r)[0]
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
                 numba_jit=False):
    """
    Run the co-clustering, Numpy-based implementation

    :param Z: m x n data matrix
    :param nclusters_row: number of row clusters
    :param nclusters_col: number of column clusters
    :param errobj: convergence threshold for the objective function
    :param niters: maximum number of iterations
    :param epsilon: numerical parameter, avoids zero arguments in log
    :return: has converged, number of iterations performed, final row and
    column clustering, error value
    """
    [m, n] = Z.shape

    if low_memory:
        row_clusters = _initialize_clusters(m, nclusters_row, low_memory=True)
        col_clusters = _initialize_clusters(n, nclusters_col, low_memory=True)
    else:
        R = _initialize_clusters(m, nclusters_row)
        C = _initialize_clusters(n, nclusters_col)

    e, old_e = 2 * errobj, 0
    s = 0
    converged = False

    Gavg = Z.mean()

    while (not converged) & (s < niters):
        # Calculate cluster based averages
        if low_memory:
            if numba_jit:
                CoCavg = (_cluster_dot_numba(Z, row_clusters, col_clusters,
                                             nclusters_row, nclusters_col) +
                          Gavg * epsilon) / (_cluster_dot_numba(np.ones(
                            (m, n)), row_clusters, col_clusters, nclusters_row,
                            nclusters_col) + epsilon)

            else:
                CoCavg = (_cluster_dot(Z, row_clusters, col_clusters,
                                       nclusters_row, nclusters_col) +
                          Gavg * epsilon) / (_cluster_dot(np.ones(
                          (m, n)), row_clusters, col_clusters, nclusters_row,
                                           nclusters_col) + epsilon)
        else:
            CoCavg = (np.dot(np.dot(R.T, Z), C) +
                      Gavg * epsilon) / (np.dot(np.dot(R.T, np.ones(
                          (m, n))), C) + epsilon)

        # Calculate distance based on row approximation
        if low_memory:
            if numba_jit:
                d_row = _distance_lowmem_numba(Z, col_clusters, CoCavg.T,
                                               epsilon)
            else:
                d_row = _distance_lowmem(Z, col_clusters, CoCavg.T, epsilon)
        else:
            d_row = _distance(Z, np.ones((m, n)), np.dot(C, CoCavg.T), epsilon)
        # Assign to best row cluster
        row_clusters = np.argmin(d_row, axis=1)
        if not low_memory:
            R = np.eye(nclusters_row, dtype=np.int32)[row_clusters]

        # Calculate distance based on column approximation
        if low_memory:
            if numba_jit:
                d_col = _distance_lowmem(Z.T, row_clusters, CoCavg, epsilon)
            else:
                d_col = _distance_lowmem_numba(Z.T, row_clusters, CoCavg,
                                               epsilon)
        else:
            d_col = _distance(Z.T, np.ones((n, m)), np.dot(R, CoCavg), epsilon)
        # Assign to best column cluster
        col_clusters = np.argmin(d_col, axis=1)
        if not low_memory:
            C = np.eye(nclusters_col, dtype=np.int32)[col_clusters]

        # Error value (actually just the column components really)
        old_e = e
        minvals_da = np.min(d_col, axis=1)
        # power 1 divergence, power 2 euclidean
        e = np.sum(np.power(minvals_da, 1))

        converged = abs(e - old_e) < errobj
        s = s + 1

    return converged, s, row_clusters, col_clusters, e
