import logging

import numpy as np

logger = logging.getLogger(__name__)


def _distance(Z, Y):
    """ Distance function """
    return Y.sum(axis=(1, 2)) - np.einsum('ijk,ljk->il', Z, np.log(Y))


def _initialize_clusters(n_el, n_clusters):
    """ Initialize cluster array """
    cluster_idx = np.mod(np.arange(n_el), n_clusters)
    return np.random.permutation(cluster_idx)


def _setup_cluster_matrix(cluster_labels, cluster_idx):
    """ Set cluster occupation matrix """
    return np.equal.outer(cluster_idx, cluster_labels)


def triclustering(Z, nclusters_row, nclusters_col, nclusters_bnd, errobj,
                  niters, row_clusters_init=None, col_clusters_init=None,
                  bnd_clusters_init=None):
    """
    Run the tri-clustering analysis, Numpy-based implementation.

    :param Z: Data array for which to run the tri-clustering analysis, with
        shape (`band`, `row`, `column`).
    :type Z: numpy.ndarray
    :param nclusters_row: Number of row clusters.
    :type nclusters_row: int
    :param nclusters_col: Number of column clusters.
    :type nclusters_col: int
    :param nclusters_bnd: Number of band clusters.
    :type nclusters_bnd: int
    :param errobj: Convergence threshold for the objective function.
    :type errobj: float, optional
    :param niters: Maximum number of iterations.
    :type niters: int, optional
    :param row_clusters_init: Initial row cluster assignment.
    :type row_clusters_init: numpy.ndarray or array_like, optional
    :param col_clusters_init: Initial column cluster assignment.
    :type col_clusters_init: numpy.ndarray or array_like, optional
    :param bnd_clusters_init: Initial band cluster assignment.
    :type bnd_clusters_init: numpy.ndarray or array_like, optional
    :return: Has converged, number of iterations performed, final row, column,
    and band clustering, approximation error of the tri-clustering.
    :type: tuple
    """
    [d, m, n] = Z.shape

    # Initialize cluster assignments
    row_clusters = row_clusters_init if row_clusters_init is not None \
        else _initialize_clusters(m, nclusters_row)
    col_clusters = col_clusters_init if col_clusters_init is not None \
        else _initialize_clusters(n, nclusters_col)
    bnd_clusters = bnd_clusters_init if bnd_clusters_init is not None \
        else _initialize_clusters(d, nclusters_bnd)

    e, old_e = 2 * errobj, 0
    s = 0
    converged = False

    while (not converged) & (s < niters):
        logger.debug(f'Iteration # {s} ..')
        # Calculate number of elements in each tri-cluster
        nel_row_clusters = np.bincount(row_clusters, minlength=nclusters_row)
        nel_col_clusters = np.bincount(col_clusters, minlength=nclusters_col)
        nel_bnd_clusters = np.bincount(bnd_clusters, minlength=nclusters_bnd)
        row_cluster_labels, = nel_row_clusters.nonzero()
        col_cluster_labels, = nel_col_clusters.nonzero()
        bnd_cluster_labels, = nel_bnd_clusters.nonzero()
        logger.debug(
            'num of populated clusters: row {}, col {}, bnd {}'.format(
                len(row_cluster_labels),
                len(col_cluster_labels),
                len(bnd_cluster_labels),
            )
        )
        nel_clusters = np.einsum(
            'i,j->ij',
            nel_row_clusters[row_cluster_labels],
            nel_col_clusters[col_cluster_labels]
        )
        nel_clusters = np.einsum(
            'i,jk->ijk',
            nel_bnd_clusters[bnd_cluster_labels],
            nel_clusters
        )

        R = _setup_cluster_matrix(row_cluster_labels, row_clusters)
        C = _setup_cluster_matrix(col_cluster_labels, col_clusters)
        B = _setup_cluster_matrix(bnd_cluster_labels, bnd_clusters)

        # calculate tri-cluster averages
        # first sum values in each tri-cluster ..
        TriCavg = np.einsum('ij,ilm->jlm', B, Z)  # .. along band axis
        TriCavg = np.einsum('ij,kim->kjm', R, TriCavg)  # .. along row axis
        TriCavg = np.einsum('ij,kli->klj', C, TriCavg)  # .. along col axis
        # finally divide by number of elements in each tri-cluster
        TriCavg = TriCavg / nel_clusters

        # unpack tri-cluster averages ..
        avg_unpck = np.einsum('ij,jkl->ikl', B, TriCavg)  # .. along band axis
        avg_unpck = np.einsum('ij,klj->kli', C, avg_unpck)  # .. along col axis
        # use these for the row cluster assignment
        idx = (1, 0, 2)
        d_row = _distance(Z.transpose(idx), avg_unpck.transpose(idx))
        row_clusters = np.argmin(d_row, axis=1)

        # unpack tri-cluster averages ..
        avg_unpck = np.einsum('ij,jkl->ikl', B, TriCavg)  # .. along band axis
        avg_unpck = np.einsum('ij,kjl->kil', R, avg_unpck)  # .. along row axis
        # use these for the col cluster assignment
        idx = (2, 0, 1)
        d_col = _distance(Z.transpose(idx), avg_unpck.transpose(idx))
        col_clusters = np.argmin(d_col, axis=1)

        # unpack tri-cluster averages ..
        avg_unpck = np.einsum('ij,kjl->kil', R, TriCavg)  # .. along row axis
        avg_unpck = np.einsum('ij,klj->kli', C, avg_unpck)  # .. along col axis
        # use these for the band cluster assignment
        d_bnd = _distance(Z, avg_unpck)
        bnd_clusters = np.argmin(d_bnd, axis=1)

        row_clusters = np.take(row_cluster_labels, row_clusters)
        col_clusters = np.take(col_cluster_labels, col_clusters)
        bnd_clusters = np.take(bnd_cluster_labels, bnd_clusters)

        # Error value (actually just the band component really)
        old_e = e
        minvals = np.min(d_bnd, axis=1)
        # power 1 divergence, power 2 euclidean
        e = np.sum(np.power(minvals, 1))

        logger.debug(f'Error = {e:+.15e}, dE = {e - old_e:+.15e}')
        converged = abs(e - old_e) < errobj
        s = s + 1
    if converged:
        logger.debug(f'Triclustering converged in {s} iterations')
    else:
        logger.debug(f'Triclustering not converged in {s} iterations')
    return converged, s, row_clusters, col_clusters, bnd_clusters, e
