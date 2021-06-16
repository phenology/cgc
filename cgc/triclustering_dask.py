import logging

import numpy as np

import dask.array as da

from dask.distributed import get_client

logger = logging.getLogger(__name__)


def _distance(Z, Y, epsilon):
    """ Distance function """
    Y = Y + epsilon
    return Y.sum(axis=(1, 2)) - da.einsum('ijk,ljk->il', Z, da.log(Y))


def _initialize_clusters(n_el, n_clusters, chunks=None):
    """ Initialize cluster array """
    cluster_idx = da.mod(da.arange(n_el, chunks=(chunks or n_el)), n_clusters)
    return da.random.permutation(cluster_idx)


def _setup_cluster_matrix(n_clusters, cluster_idx):
    """ Set cluster occupation matrix """
    return da.eye(n_clusters, dtype=np.bool, chunks=n_clusters)[cluster_idx]


def triclustering(Z, nclusters_row, nclusters_col, nclusters_bnd, errobj,
                  niters, epsilon, row_clusters_init=None,
                  col_clusters_init=None, bnd_clusters_init=None):
    """
    Run the tri-clustering, Dask implementation

    :param Z: d x m x n data matrix
    :param nclusters_row: number of row clusters
    :param nclusters_col: number of column clusters
    :param nclusters_bnd: number of band clusters
    :param errobj: convergence threshold for the objective function
    :param niters: maximum number of iterations
    :param epsilon: numerical parameter, avoids zero arguments in log
    :param row_clusters_init: initial row cluster assignment
    :param col_clusters_init: initial column cluster assignment
    :param bnd_clusters_init: initial column cluster assignment
    :return: has converged, number of iterations performed. final row,
    column, and band clustering, error value
    """
    client = get_client()

    Z = da.array(Z) if not isinstance(Z, da.Array) else Z

    [d, m, n] = Z.shape
    bnd_chunks, row_chunks, col_chunks = Z.chunksize

    row_clusters = da.array(row_clusters_init) \
        if row_clusters_init is not None \
        else _initialize_clusters(m, nclusters_row, chunks=row_chunks)
    col_clusters = da.array(col_clusters_init) \
        if col_clusters_init is not None \
        else _initialize_clusters(n, nclusters_col, chunks=col_chunks)
    bnd_clusters = da.array(bnd_clusters_init) \
        if bnd_clusters_init is not None \
        else _initialize_clusters(d, nclusters_bnd, chunks=bnd_chunks)
    R = _setup_cluster_matrix(nclusters_row, row_clusters)
    C = _setup_cluster_matrix(nclusters_col, col_clusters)
    B = _setup_cluster_matrix(nclusters_bnd, bnd_clusters)

    e, old_e = 2 * errobj, 0
    s = 0
    converged = False

    Gavg = Z.mean()

    while (not converged) & (s < niters):
        logger.debug(f'Iteration # {s} ..')
        # Calculate number of elements in each tri-cluster
        nel_row_clusters = da.bincount(row_clusters, minlength=nclusters_row)
        nel_col_clusters = da.bincount(col_clusters, minlength=nclusters_col)
        nel_bnd_clusters = da.bincount(bnd_clusters, minlength=nclusters_bnd)
        logger.debug(
            'num of populated clusters: row {}, col {}, bnd {}'.format(
                da.sum(nel_row_clusters > 0).compute(),
                da.sum(nel_col_clusters > 0).compute(),
                da.sum(nel_bnd_clusters > 0).compute()
            )
        )
        nel_clusters = da.einsum('i,j->ij', nel_row_clusters, nel_col_clusters)
        nel_clusters = da.einsum('i,jk->ijk', nel_bnd_clusters, nel_clusters)

        # calculate tri-cluster averages (epsilon takes care of empty clusters)
        # first sum values in each tri-cluster ..
        TriCavg = da.einsum('ij,ilm->jlm', B, Z)  # .. along band axis
        TriCavg = da.einsum('ij,kim->kjm', R, TriCavg)  # .. along row axis
        TriCavg = da.einsum('ij,kli->klj', C, TriCavg)  # .. along col axis
        # finally divide by number of elements in each tri-cluster
        TriCavg = (TriCavg + Gavg * epsilon) / (nel_clusters + epsilon)


        # unpack tri-cluster averages ..
        avg_unpck = da.einsum('ij,jkl->ikl', B, TriCavg)  # .. along band axis
        avg_unpck = da.einsum('ij,klj->kli', C, avg_unpck)  # .. along col axis
        # use these for the row cluster assignment
        idx = (1, 0, 2)
        d_row = _distance(Z.transpose(idx), avg_unpck.transpose(idx), epsilon)
        row_clusters = da.argmin(d_row, axis=1)
        R = _setup_cluster_matrix(nclusters_row, row_clusters)

        # TriCavg = da.einsum('ij,ilm->jlm', B, Z)  # .. along band axis
        # TriCavg = da.einsum('ij,kim->kjm', R, TriCavg)  # .. along row axis
        # TriCavg = da.einsum('ij,kli->klj', C, TriCavg)  # .. along col axis
        # # finally divide by number of elements in each tri-cluster
        # TriCavg = (TriCavg + Gavg * epsilon) / (nel_clusters + epsilon)

        # unpack tri-cluster averages ..
        avg_unpck = da.einsum('ij,jkl->ikl', B, TriCavg)  # .. along band axis
        avg_unpck = da.einsum('ij,kjl->kil', R, avg_unpck)  # .. along row axis
        # use these for the col cluster assignment
        idx = (2, 0, 1)
        d_col = _distance(Z.transpose(idx), avg_unpck.transpose(idx), epsilon)
        col_clusters = da.argmin(d_col, axis=1)
        C = _setup_cluster_matrix(nclusters_col, col_clusters)

        # TriCavg = da.einsum('ij,ilm->jlm', B, Z)  # .. along band axis
        # TriCavg = da.einsum('ij,kim->kjm', R, TriCavg)  # .. along row axis
        # TriCavg = da.einsum('ij,kli->klj', C, TriCavg)  # .. along col axis
        # # finally divide by number of elements in each tri-cluster
        # TriCavg = (TriCavg + Gavg * epsilon) / (nel_clusters + epsilon)

        # unpack tri-cluster averages ..
        avg_unpck = da.einsum('ij,kjl->kil', R, TriCavg)  # .. along row axis
        avg_unpck = da.einsum('ij,klj->kli', C, avg_unpck)  # .. along col axis
        # use these for the band cluster assignment
        d_bnd = _distance(Z, avg_unpck, epsilon)
        bnd_clusters = da.argmin(d_bnd, axis=1)
        B = _setup_cluster_matrix(nclusters_bnd, bnd_clusters)

        # Error value (actually just the band component really)
        old_e = e
        minvals = da.min(d_bnd, axis=1)
        # power 1 divergence, power 2 euclidean
        e = da.sum(da.power(minvals, 1))
        row_clusters, R, col_clusters, C, bnd_clusters, B, e = client.persist(
            [row_clusters, R, col_clusters, C, bnd_clusters, B, e]
        )
        e = e.compute()
        logger.debug(f'Error = {e:+.15e}, dE = {e - old_e:+.15e}')
        converged = abs(e - old_e) < errobj
        s = s + 1
    if converged:
        logger.debug(f'Triclustering converged in {s} iterations')
    else:
        logger.debug(f'Triclustering not converged in {s} iterations')
    return converged, s, row_clusters, col_clusters, bnd_clusters, e
