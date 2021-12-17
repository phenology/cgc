import logging

import dask.array as da
from dask.distributed import get_client, rejoin, secede

logger = logging.getLogger(__name__)


def _distance(Z, Y):
    """ Distance function """
    # The first term below is equal to one row of: da.dot(da.ones(m, n), Y)
    # with Z.shape = (m, n) and Y.shape = (n, k)
    return Y.sum(axis=0, keepdims=True) - da.matmul(Z, da.log(Y))


def _initialize_clusters(n_el, n_clusters, chunks=None):
    """ Initialize cluster array """
    cluster_idx = da.mod(da.arange(n_el, chunks=(chunks or n_el)), n_clusters)
    return da.random.permutation(cluster_idx)


def _setup_cluster_matrix(cluster_labels, cluster_idx):
    """ Set cluster occupation matrix """
    return da.equal.outer(cluster_idx, cluster_labels)


def coclustering(Z, nclusters_row, nclusters_col, errobj, niters,
                 col_clusters_init=None, row_clusters_init=None,
                 run_on_worker=False):
    """
    Run the co-clustering analysis, Dask implementation.

    :param Z: Data matrix for which to run the co-clustering analysis
    :type Z: dask.array.Array or array_like
    :param nclusters_row: Number of row clusters.
    :type nclusters_row: int
    :param nclusters_col: Number of column clusters.
    :type nclusters_col: int
    :param errobj: Convergence threshold for the objective function.
    :type errobj: float, optional
    :param niters: Maximum number of iterations.
    :type niters: int, optional
    :param row_clusters_init: Initial row cluster assignment.
    :type row_clusters_init: numpy.ndarray or array_like, optional
    :param col_clusters_init: Initial column cluster assignment.
    :type col_clusters_init: numpy.ndarray or array_like, optional
    :param run_on_worker: Whether the function is submitted to a Dask worker
    :type run_on_worker: bool, optional
    :return: Has converged, number of iterations performed, final row and
    column clustering, approximation error of the co-clustering.
    :type: tuple
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

    e, old_e = 2 * errobj, 0
    s = 0
    converged = False

    while (not converged) & (s < niters):
        logger.debug(f'Iteration # {s} ..')
        # Calculate cluster based averages
        # nel_clusters is a matrix with the number of elements per co-cluster
        # originally computed as:  da.dot(da.dot(R.T, da.ones((m, n))), C)
        nel_row_clusters = da.bincount(row_clusters, minlength=nclusters_row)
        nel_col_clusters = da.bincount(col_clusters, minlength=nclusters_col)
        row_cluster_labels = nel_row_clusters.nonzero()[0].compute()
        col_cluster_labels = nel_col_clusters.nonzero()[0].compute()
        logger.debug('num of populated clusters: row {}, col {}'.format(
                        len(row_cluster_labels),
                        len(col_cluster_labels)))
        R = _setup_cluster_matrix(row_cluster_labels, row_clusters)
        C = _setup_cluster_matrix(col_cluster_labels, col_clusters)
        nel_clusters = da.outer(nel_row_clusters[row_cluster_labels],
                                nel_col_clusters[col_cluster_labels])
        CoCavg = da.matmul(da.matmul(R.T, Z), C) / nel_clusters

        # Calculate distance based on row approximation
        d_row = _distance(Z, da.matmul(C, CoCavg.T))
        # Assign to best row cluster
        row_clusters = da.argmin(d_row, axis=1)

        # Calculate distance based on column approximation
        d_col = _distance(Z.T, da.matmul(R, CoCavg))
        # Assign to best column cluster
        col_clusters = da.argmin(d_col, axis=1)

        # Error value (actually just the column components really)
        old_e = e
        minvals = da.min(d_col, axis=1)
        # power 1 divergence, power 2 euclidean
        e = da.sum(da.power(minvals, 1))
        row_clusters, col_clusters, e = client.persist([
            da.take(row_cluster_labels, row_clusters),
            da.take(col_cluster_labels, col_clusters),
            e
        ])
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
