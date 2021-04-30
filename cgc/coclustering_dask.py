import logging

import numpy as np

import dask.array as da
from dask.distributed import get_client, rejoin, secede

logger = logging.getLogger(__name__)


def _distance(Z, Y, epsilon):
    """ Distance function """
    Y = Y + epsilon
    # The first term below is equal to one row of: da.dot(da.ones(m, n), Y)
    # with Z.shape = (m, n) and Y.shape = (n, k)
    return Y.sum(axis=0, keepdims=True) - da.matmul(Z, da.log(Y))


def _initialize_clusters(n_el, n_clusters, chunks=None):
    """ Initialize cluster array """
    cluster_idx = da.mod(da.arange(n_el, chunks=(chunks or n_el)), n_clusters)
    return da.random.permutation(cluster_idx)


def _setup_cluster_matrix(n_clusters, cluster_idx):
    """ Set cluster occupation matrix """
    return da.eye(n_clusters, dtype=np.bool, chunks=n_clusters)[cluster_idx]


def coclustering(Z, nclusters_row, nclusters_col, errobj, niters, epsilon,
                 col_clusters_init=None, row_clusters_init=None,
                 run_on_worker=False):
    """
    Run the co-clustering, Dask implementation

    :param Z: m x n data matrix
    :param nclusters_row: num row clusters
    :param nclusters_col: number of column clusters
    :param errobj: convergence threshold for the objective function
    :param niters: maximum number of iterations
    :param epsilon: numerical parameter, avoids zero arguments in log
    :param row_clusters_init: initial row cluster assignment
    :param col_clusters_init: initial column cluster assignment
    :param run_on_worker: whether the function is submitted to a Dask worker
    :return: has converged, number of iterations performed. final row and
    column clustering, error value
    """
    client = get_client()

    Z = da.array(Z) if not isinstance(Z, da.Array) else Z

    [m, n] = Z.shape
    row_chunks, col_chunks = Z.chunksize

    row_clusters = da.array(row_clusters_init) \
        if row_clusters_init is not None \
        else _initialize_clusters(m, nclusters_row, chunks=row_chunks)
    col_clusters = da.array(col_clusters_init) \
        if col_clusters_init is not None \
        else _initialize_clusters(n, nclusters_col, chunks=col_chunks)
    R = _setup_cluster_matrix(nclusters_row, row_clusters)
    C = _setup_cluster_matrix(nclusters_col, col_clusters)

    e, old_e = 2 * errobj, 0
    s = 0
    converged = False

    Gavg = Z.mean()

    while (not converged) & (s < niters):
        logger.debug(f'Iteration # {s} ..')
        # Calculate cluster based averages
        # nel_clusters is a matrix with the number of elements per co-cluster
        # originally computed as:  da.dot(da.dot(R.T, da.ones((m, n))), C)
        nel_row_clusters = da.bincount(row_clusters, minlength=nclusters_row)
        nel_col_clusters = da.bincount(col_clusters, minlength=nclusters_col)
        logger.debug('num of populated clusters: row {}, col {}'.format(
                        da.sum(nel_row_clusters > 0).compute(),
                        da.sum(nel_col_clusters > 0).compute()))
        nel_clusters = da.outer(nel_row_clusters, nel_col_clusters)
        CoCavg = (da.matmul(da.matmul(R.T, Z), C) + Gavg * epsilon) / \
                 (nel_clusters + epsilon)

        # Calculate distance based on row approximation
        d_row = _distance(Z, da.matmul(C, CoCavg.T), epsilon)
        # Assign to best row cluster
        row_clusters = da.argmin(d_row, axis=1)
        R = _setup_cluster_matrix(nclusters_row, row_clusters)

        # Calculate distance based on column approximation
        d_col = _distance(Z.T, da.matmul(R, CoCavg), epsilon)
        # Assign to best column cluster
        col_clusters = da.argmin(d_col, axis=1)
        C = _setup_cluster_matrix(nclusters_col, col_clusters)

        # Error value (actually just the column components really)
        old_e = e
        minvals = da.min(d_col, axis=1)
        # power 1 divergence, power 2 euclidean
        e = da.sum(da.power(minvals, 1))
        row_clusters, R, col_clusters, C, e = client.persist([row_clusters, R,
                                                              col_clusters, C,
                                                              e])
        if run_on_worker:
            # this is workaround for e.compute() for a function that runs
            # on a worker with multiple threads
            # https://github.com/dask/distributed/issues/3827
            e = client.compute(e)
            secede()
            e = e.result()
            rejoin()
        else:
            e = e.compute()
        logger.debug(f'Error = {e:+.15e}, dE = {e - old_e:+.15e}')
        converged = abs(e - old_e) < errobj
        s = s + 1
    if converged:
        logger.debug(f'Coclustering converged in {s} iterations')
    else:
        logger.debug(f'Coclustering not converged in {s} iterations')
    return converged, s, row_clusters, col_clusters, e
