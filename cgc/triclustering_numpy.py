import logging

import numpy as np

logger = logging.getLogger(__name__)


def _distance(Z, X, Y, epsilon):
    """ Distance function """
    Y = Y + epsilon
    d = np.dot(X, Y) - np.dot(Z, np.log(Y))
    return d


def _initialize_clusters(n_el, n_clusters):
    """ Initialize cluster array """
    cluster_idx = np.mod(np.arange(n_el), n_clusters)
    return np.random.permutation(cluster_idx)


def _setup_cluster_matrix(n_clusters, cluster_idx):
    """ Set cluster occupation matrix """
    return np.eye(n_clusters, dtype=np.int32)[cluster_idx]


def triclustering(Z, nclusters_row, nclusters_col, nclusters_bnd, errobj,
                  niters, epsilon, row_clusters_init=None,
                  col_clusters_init=None, bnd_clusters_init=None):
    """
    Run the tri-clustering, Numpy-based implementation

    :param Z: d x m x n data matrix
    :param nclusters_row: number of row clusters
    :param nclusters_col: number of column clusters
    :param nclusters_bnd: number of band clusters
    :param errobj: convergence threshold for the objective function
    :param niters: maximum number of iterations
    :param epsilon: numerical parameter, avoids zero arguments in log
    :param row_clusters_init: initial row cluster assignment
    :param col_clusters_init: initial column cluster assignment
    :param bnd_clusters_init: initial band cluster assignment
    :return: has converged, number of iterations performed, final row,
    column, and band clustering, error value
    """
    [d, m, n] = Z.shape

    # Setup matrices to ..
    Y = np.concatenate(Z, axis=1)  # .. update rows
    Y1 = np.concatenate(Z, axis=0)  # .. update columns
    Y2 = Z.reshape(d, m*n)  # .. update bands

    # Calculate average
    Gavg = Y.mean()

    # Initialize cluster assignments
    row_clusters = row_clusters_init if row_clusters_init is not None \
        else _initialize_clusters(m, nclusters_row)
    col_clusters = col_clusters_init if col_clusters_init is not None \
        else _initialize_clusters(n, nclusters_col)
    bnd_clusters = bnd_clusters_init if bnd_clusters_init is not None \
        else _initialize_clusters(d, nclusters_bnd)
    x_clusters = np.tile(bnd_clusters, n) if bnd_clusters_init is not None \
        else _initialize_clusters(n*d, nclusters_bnd)
    R = _setup_cluster_matrix(nclusters_row, row_clusters)
    C = _setup_cluster_matrix(nclusters_col, col_clusters)
    B = _setup_cluster_matrix(nclusters_bnd, bnd_clusters)
    C1 = _setup_cluster_matrix(nclusters_bnd, x_clusters)

    e, old_e = 2 * errobj, 0
    s = 0
    converged = False

    while (not converged) & (s < niters):
        logger.debug(f'Iteration # {s} ..')
        # Obtain all the cluster based averages
        CoCavg = (np.dot(np.dot(R.T, Y), C1) + Gavg * epsilon) / (
                np.dot(np.dot(R.T, np.ones((m, n * d))), C1) + epsilon)

        # Calculate distance based on row approximation
        d2 = _distance(Y, np.ones((m, n * d)), np.dot(C1, CoCavg.T), epsilon)

        # Assign to best row cluster
        row_clusters = np.argmin(d2, axis=1)
        R = _setup_cluster_matrix(nclusters_row, row_clusters)
        R1 = np.tile(R, (d, 1))

        # Obtain all the cluster based averages
        CoCavg1 = (np.dot(np.dot(R1.T, Y1), C) + Gavg * epsilon) / (
                np.dot(np.dot(R1.T, np.ones((m * d, n))), C) + epsilon)

        # Calculate distance based on column approximation
        d2 = _distance(Y1.T, np.ones((n, m * d)), np.dot(R1, CoCavg1), epsilon)

        # Assign to best column cluster
        col_clusters = np.argmin(d2, axis=1)
        C = _setup_cluster_matrix(nclusters_col, col_clusters)
        C2 = np.tile(C, (m, 1))

        # Obtain all the cluster based averages
        CoCavg2 = (np.dot(np.dot(B.T, Y2), C2) + Gavg * epsilon) / (
                np.dot(np.dot(B.T, np.ones((d, m * n))), C2) + epsilon)

        # Calculate distance based on band approximation
        d2 = _distance(Y2, np.ones((d, m * n)), np.dot(C2, CoCavg2.T), epsilon)

        # Assign to best band cluster
        bnd_clusters = np.argmin(d2, axis=1)
        B = _setup_cluster_matrix(nclusters_bnd, bnd_clusters)
        C1 = np.tile(B, (n, 1))

        # Error value
        old_e = e
        minvals = np.amin(d2, axis=1)
        # power 1 divergence, power 2 euclidean
        e = sum(np.power(minvals, 1))

        logger.debug(f'Error = {e:+.15e}, dE = {e - old_e:+.15e}')
        converged = abs(e - old_e) < errobj
        s = s + 1
    if converged:
        logger.debug(f'Triclustering converged in {s} iterations')
    else:
        logger.debug(f'Triclustering not converged in {s} iterations')
    return converged, s, row_clusters, col_clusters, bnd_clusters, e
