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
    eye = np.eye(n_clusters, dtype=np.int32)
    return eye[cluster_idx]


def coclustering(Z, nclusters_row, nclusters_col, errobj, niters, epsilon):
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

    R = _initialize_clusters(m, nclusters_row)
    C = _initialize_clusters(n, nclusters_col)

    e, old_e = 2 * errobj, 0
    s = 0
    converged = False

    Gavg = Z.mean()

    while (not converged) & (s < niters):
        # Calculate cluster based averages
        CoCavg = (np.dot(np.dot(R.T, Z), C) + Gavg * epsilon) / (
                np.dot(np.dot(R.T, np.ones((m, n))), C) + epsilon)

        # Calculate distance based on row approximation
        d_row = _distance(Z, np.ones((m, n)), np.dot(C, CoCavg.T), epsilon)
        # Assign to best row cluster
        row_clusters = np.argmin(d_row, axis=1)
        R = np.eye(nclusters_row, dtype=np.int32)[row_clusters]

        # Calculate distance based on column approximation
        d_col = _distance(Z.T, np.ones((n, m)), np.dot(R, CoCavg), epsilon)
        # Assign to best column cluster
        col_clusters = np.argmin(d_col, axis=1)
        C = np.eye(nclusters_col, dtype=np.int32)[col_clusters]

        # Error value (actually just the column components really)
        old_e = e
        minvals_da = np.min(d_col, axis=1)
        # power 1 divergence, power 2 euclidean
        e = np.sum(np.power(minvals_da, 1))

        converged = abs(e - old_e) < errobj
        s = s + 1

    return converged, s, row_clusters, col_clusters, e
