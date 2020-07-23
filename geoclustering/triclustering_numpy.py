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


def triclustering(Z, nclusters_row, nclusters_col, nclusters_bnd, errobj,
                  niters, epsilon):
    """
    Run the tri-clustering, Numpy-based implementation

    :param Z: d x m x n data matrix
    :param nclusters_row: number of row clusters
    :param nclusters_col: number of column clusters
    :param nclusters_bnd: number of band clusters
    :param errobj: convergence threshold for the objective function
    :param niters: maximum number of iterations
    :param epsilon: numerical parameter, avoids zero arguments in log
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

    # Randomly initialize row and column clustering
    R = _initialize_clusters(m, nclusters_row)
    C1 = _initialize_clusters(n * d, nclusters_bnd)
    C = _initialize_clusters(n, nclusters_col)
    B = _initialize_clusters(d, nclusters_bnd)

    e, old_e = 2 * errobj, 0
    s = 0
    converged = False

    while (not converged) & (s < niters):
        # Obtain all the cluster based averages
        CoCavg = (np.dot(np.dot(R.T, Y), C1) + Gavg * epsilon) / (
                np.dot(np.dot(R.T, np.ones((m, n * d))), C1) + epsilon)

        # Calculate distance based on row approximation
        d2 = _distance(Y, np.ones((m, n * d)), np.dot(C1, CoCavg.T), epsilon)

        # Assign to best row cluster
        row_clusters = np.argmin(d2, axis=1)
        R = np.eye(nclusters_row)[row_clusters, :]
        R1 = np.tile(R, (d, 1))

        # Obtain all the cluster based averages
        CoCavg1 = (np.dot(np.dot(R1.T, Y1), C) + Gavg * epsilon) / (
                np.dot(np.dot(R1.T, np.ones((m * d, n))), C) + epsilon)

        # Calculate distance based on column approximation
        d2 = _distance(Y1.T, np.ones((n, m * d)), np.dot(R1, CoCavg1), epsilon)

        # Assign to best column cluster
        col_clusters = np.argmin(d2, axis=1)
        C = np.eye(nclusters_col)[col_clusters, :]
        C2 = np.tile(C, (m, 1))

        # Obtain all the cluster based averages
        CoCavg2 = (np.dot(np.dot(B.T, Y2), C2) + Gavg * epsilon) / (
                np.dot(np.dot(B.T, np.ones((d, m * n))), C2) + epsilon)

        # Calculate distance based on column approximation
        d2 = _distance(Y2, np.ones((d, m * n)), np.dot(C2, CoCavg2.T), epsilon)

        # Assign to best column cluster
        bnd_clusters = np.argmin(d2, axis=1)
        B = np.eye(nclusters_bnd)[bnd_clusters, :]
        C1 = np.tile(B, (n, 1))

        # Error value (actually just the column components really)
        old_e = e
        minvals = np.amin(d2, axis=1)
        # power 1 divergence, power 2 euclidean
        e = sum(np.power(minvals, 1))

        converged = abs(e - old_e) < errobj
        s = s + 1

    return converged, s, row_clusters, col_clusters, bnd_clusters, e
