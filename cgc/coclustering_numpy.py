import logging

import numpy as np

logger = logging.getLogger(__name__)


def _distance(Z, Y, epsilon):
    """ Distance function """
    Y = Y + epsilon
    d = np.dot(np.ones(Z.shape), Y) - np.dot(Z, np.log(Y))
    return d


def _initialize_clusters(n_el, n_clusters):
    """ Initialize cluster array """
    cluster_idx = np.mod(np.arange(n_el), n_clusters)
    return np.random.permutation(cluster_idx)


def _setup_cluster_matrix(n_clusters, cluster_idx):
    """ Set cluster occupation matrix """
    return np.eye(n_clusters, dtype=np.bool)[cluster_idx]


def coclustering(Z, nclusters_row, nclusters_col,
                 errobj, niters, epsilon,
                 row_clusters_init=None, col_clusters_init=None):
    """
    Run the co-clustering, Numpy-based implementation

    :param Z: m x n data matrix
    :param nclusters_row: number of row clusters
    :param nclusters_col: number of column clusters
    :param errobj: convergence threshold for the objective function
    :param niters: maximum number of iterations
    :param epsilon: numerical parameter, avoids zero arguments in log
    :param row_clusters_init: initial row cluster assignment
    :param col_clusters_init: initial column cluster assignment
    :return: has converged, number of iterations performed, row clusters,
             column clusters, error value
    """
    [m, n] = Z.shape

    # Set initial clusters
    row_clusters = row_clusters_init if row_clusters_init is not None \
        else _initialize_clusters(m, nclusters_row)
    col_clusters = col_clusters_init if col_clusters_init is not None \
        else _initialize_clusters(n, nclusters_col)
    R = _setup_cluster_matrix(nclusters_row, row_clusters)
    C = _setup_cluster_matrix(nclusters_col, col_clusters)

    e, old_e = 2 * errobj, 0
    s = 0
    converged = False

    Gavg = Z.mean()

    while (not converged) & (s < niters):
        logger.debug(f'Iteration # {s} ..')
        # Calculate cluster based averages
        dot = np.dot(np.dot(R.T, np.ones((m, n))), C)
        CoCavg = (np.dot(np.dot(R.T, Z), C) + Gavg * epsilon) / (dot + epsilon)

        # Calculate distance based on row approximation
        d = _distance(Z, np.dot(C, CoCavg.T), epsilon)
        # Assign to best row cluster
        row_clusters = np.argmin(d, axis=1)
        R = _setup_cluster_matrix(nclusters_row, row_clusters)

        # Calculate distance based on column approximation
        d = _distance(Z.T, np.dot(R, CoCavg), epsilon)
        # Assign to best column cluster
        col_clusters = np.argmin(d, axis=1)
        C = _setup_cluster_matrix(nclusters_col, col_clusters)

        # Error value (just the column components)
        old_e = e
        minvals = np.min(d, axis=1)
        e = np.sum(minvals)

        logger.debug(f'Error = {e:+.15e}, dE = {e - old_e:+.15e}')
        converged = abs(e - old_e) < errobj
        s = s + 1
    if converged:
        logger.debug(f'Coclustering converged in {s} iterations')
    else:
        logger.debug(f'Coclustering not converged in {s} iterations')
    return converged, s, row_clusters, col_clusters, e
