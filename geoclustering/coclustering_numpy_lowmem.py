import numpy as np


def _distance(Z, X, Y, epsilon):
    """ Distance function """
    Y = Y + epsilon
    d = np.dot(X, Y) - np.dot(Z, np.log(Y))
    return d


def _initialize_clusters(n_el, n_clusters):
    """ Initialize cluster occupation matrix """
    cluster_idx = np.mod(np.arange(n_el), n_clusters)
    cluster_idx = np.random.permutation(cluster_idx)
    return cluster_idx


def _full_matrix(cluster_idx, n_clusters):
    eye = np.eye(n_clusters, dtype=np.int32)
    return eye[cluster_idx]


def _cluster_dot(Z, row_clusters, col_clusters, k, l):
    """
    To replace np.dot(np.dot(R.T, Z), C), where R and C are full matrix
    """
    product = np.zeros((k,l))
    for r in range(0, k):
        for c in range(0, l):
            idx_r = np.where(row_clusters==r)[0]
            idx_c = np.where(col_clusters==c)[0]
            idx_rc = np.array(np.meshgrid(idx_r, idx_c)).T.reshape(-1,2)
            product[r,c] = np.sum(Z[idx_rc[:,0], idx_rc[:,1]])
    return product

def coclustering(Z, k, l, errobj, niters, epsilon):
    """
    Run the co-clustering, Numpy-based implementation

    :param Z: m x n data matrix
    :param k: num row clusters
    :param l: num col clusters
    :param errobj: precision of obj fun for convergence
    :param niters: max iterations
    :param epsilon: precision of matrix elements
    :return: has converged, final row clustering, final column clustering,
    error value
    """
    [m, n] = Z.shape

    row_clusters = _initialize_clusters(m, k)
    col_clusters = _initialize_clusters(n, l)

    e, old_e = 2 * errobj, 0
    s = 1

    Gavg = Z.mean()

    while (abs(e - old_e) > errobj) & (s <= niters):
        # Calculate cluster based averages
        CoCavg = (_cluster_dot(Z, row_clusters, col_clusters, k, l) + Gavg * epsilon) / (
                _cluster_dot(np.ones((m, n)), row_clusters, col_clusters, k, l) + epsilon)

        # Calculate distance based on row approximation
        d_row = _distance(Z, np.ones((m, n)), np.dot(_full_matrix(col_clusters, l), CoCavg.T), epsilon)
        # Assign to best row cluster
        row_clusters = np.argmin(d_row, axis=1)
        R = np.eye(k, dtype=np.int32)[row_clusters]

        # Calculate distance based on column approximation
        d_col = _distance(Z.T, np.ones((n, m)), np.dot(R, CoCavg), epsilon)
        # Assign to best column cluster
        col_clusters = np.argmin(d_col, axis=1)
        C = np.eye(l, dtype=np.int32)[col_clusters]

        # Error value (actually just the column components really)
        old_e = e
        minvals_da = np.min(d_col, axis=1)
        # power 1 divergence, power 2 euclidean
        e = np.sum(np.power(minvals_da, 1))

        s = s + 1

    converged = s <= niters
    return converged, s, row_clusters, col_clusters, e
