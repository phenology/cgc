import math
import logging
import numpy as np


def mem_estimate_coclustering_numpy(n_rows,
                                    n_cols,
                                    nclusters_row,
                                    nclusters_col,
                                    out_unit=None):
    """
    Estimate the maximum memory usage of `cgc.coclustering_numpy`, given the
    matrix size (`n_rows`, `n_cols`) and the number of row/column clusters
    (`nclusters_row`, `nclusters_col`).

    The estimated memory usage is the sum of the size of all major arrays
    simultaneously allocated in `cgc.coclustering_numpy.coclustering`.

    Depending on the shape of the data matrix, there are two possible memory
    peaks, corresponding to either the first or the second call to
    `cgc.coclustering_numpy._distance()`.

    :param n_rows: Number of rows in the data matrix.
    :type n_rows: int
    :param n_cols: Number of columns in the data matrix.
    :type n_cols: int
    :param nclusters_row: Number of row clusters.
    :type nclusters_row: int
    :param nclusters_col: Number of column clusters.
    :type nclusters_col: int
    :param out_unit: Output units, choose between "B", "KB", "MB", "GB"
    :type out_unit: str
    :return: Estimated memory usage, unit, peak.
    :type: tuple
    """
    logger = logging.getLogger(__name__)

    # Size of major matrix
    Z = _est_arr_size((n_rows, n_cols))
    C = _est_arr_size((n_cols, nclusters_col), 1)
    R = _est_arr_size((n_rows, nclusters_row), 1)

    # First call _distance
    Y1 = _est_arr_size((n_cols, nclusters_row))
    d1 = _est_arr_size(
        (n_rows,
         nclusters_row)) * 2  # x2 because "Y.sum(axis=0, keepdims=True)"
    # is also exanpded to (n_rows,nclusters_row) for matrix minus
    # operation

    # Second call _distance
    Y2 = _est_arr_size((n_rows, nclusters_col))
    d2 = _est_arr_size(
        (n_cols,
         nclusters_col)) * 2  # x2 because "Y.sum(axis=0, keepdims=True)"
    # is also exanpded to (n_rows,nclusters_row) for matrix minus
    # operation

    # Compare two potential peaks
    mem1 = Z + C + R + Y1 + d1
    mem2 = Z + C + R + Y2 + d2
    mem_usage = max([mem1, mem2])
    peak = [mem1, mem2].index(mem_usage) + 1

    # Make results human redable
    mem_usage, unit = _human_size(mem_usage, out_unit)

    logger.info('Estimated memory usage: {:.2f}{}, peak number: {}'.format(
        mem_usage, unit, peak))

    return mem_usage, unit, peak


def _est_arr_size(shape, nbytes=8):
    """
    Estimate size of an n-D Numpy array

    :param shape: shape of the n-D Numpy array
    :type shape: tuple or list
    :param nbytes: size in bytes of a single element, defaults to 8 (float64)
    :return: Size of the array in bytes
    """

    return np.prod(shape, dtype="int64") * nbytes


def _human_size(size_bytes, out_unit=None):
    """
    Convert size in bytes to human readable format

    :param size_bytes: size in bytes
    :return: human readable size, unit
    """
    if size_bytes == 0:
        return "0B"
    unit = ("B", "KB", "MB", "GB")
    if out_unit is None:
        i = min(int(math.floor(math.log(size_bytes, 1024))),
                3)  # no more than 3
    else:
        assert out_unit in unit, "Unknow size unit {}".format(out_unit)
        i = unit.index(out_unit)

    p = math.pow(1024, i)
    s = size_bytes / p

    return s, unit[i]


def calculate_cocluster_averages(Z, row_clusters, col_clusters,
                                 nclusters_row=None, nclusters_col=None):
    """
    Calculate the co-cluster averages from the data array and the row- and
    column-cluster assignments.

    :param Z: Data matrix.
    :type Z: numpy.ndarray or dask.array.Array
    :param row_clusters: Row cluster assignment.
    :type row_clusters: numpy.ndarray or array_like
    :param col_clusters: Column cluster assignment.
    :type col_clusters: numpy.ndarray or array_like
    :param nclusters_row: Number of row clusters. If not provided, it is set as
        the number of unique elements in `row_clusters`.
    :type nclusters_row: int, optional
    :param nclusters_col: Number of column clusters. If not provided, it is set
        as the number of unique elements in `col_clusters`.
    :type nclusters_col: int, optional
    :return: Array with co-cluster averages, shape
        (`nclusters_row`, `nclusters_col`). Elements corresponding to empty
        co-clusters are set as NaN.
    :type: numpy.ndarray
    """
    return calculate_cluster_feature(
        Z,
        np.mean,
        (row_clusters, col_clusters),
        (nclusters_row, nclusters_col)
    )


def calculate_tricluster_averages(Z, row_clusters, col_clusters, bnd_clusters,
                                  nclusters_row=None, nclusters_col=None,
                                  nclusters_bnd=None):
    """
    Calculate the tri-cluster averages from the data array and the band-,
    row- and column-cluster assignments.

    :param Z: Data array, with shape (`bands`, `rows`, `columns`).
    :type Z: numpy.ndarray or dask.array.Array
    :param row_clusters: Row cluster assignment.
    :type row_clusters: numpy.ndarray or array_like
    :param col_clusters: Column cluster assignment.
    :type col_clusters: numpy.ndarray or array_like
    :param bnd_clusters: Band cluster assignment.
    :type bnd_clusters: numpy.ndarray or array_like
    :param nclusters_row: Number of row clusters. If not provided, it is set as
        the number of unique elements in `row_clusters`.
    :type nclusters_row: int, optional
    :param nclusters_col: Number of column clusters. If not provided, it is set
        as the number of unique elements in `col_clusters`.
    :type nclusters_col: int, optional
    :param nclusters_bnd: Number of band clusters. If not provided, it is set
        as the number of unique elements in `col_clusters`.
    :type nclusters_bnd: int, optional
    :return: Array with tri-cluster averages, shape
        (`nclusters_bnd`, `nclusters_row`, `nclusters_col`). Elements
        corresponding to empty tri-clusters are set as NaN.
    :type: numpy.ndarray
    """
    return calculate_cluster_feature(
        Z,
        np.mean,
        (bnd_clusters, row_clusters, col_clusters),
        (nclusters_bnd, nclusters_row, nclusters_col)
    )


def calculate_cluster_feature(Z, function, clusters, nclusters=None, **kwargs):
    """
    Calculate features for clusters. This function works in N dimensions
    (N=2, 3, ...) making it suitable to calculate features for both co-clusters
    and tri-clusters.

    :param Z: Data array (N dimensions).
    :type Z: numpy.ndarray or dask.array.Array
    :param function: Function to run over the cluster elements to calculate the
        desired feature. Should take as an input a N-dimensional array and
        return a scalar.
    :type function: Callable
    :param clusters: Iterable with length N. It should contain the cluster
        labels for each dimension, following the same ordering as for Z
    :type clusters: tuple, list, or numpy.ndarray
    :param nclusters: Iterable with length N. It should contains the number of
        clusters in each dimension, following the same ordering as for Z.  If
        not provided, it is set as the number of unique elements in each
        dimension.
    :type nclusters: tuple, list, or numpy.ndarray, optional
    :param kwargs: keyword arguments to be passed to the input function
        together with the input data array for each cluster
    :type kwargs: dict, optional
    :return: the desired feature is computed for each cluster and added to an
        array with N dimensions. It has dimension N and shape equal to
        nclusters.
    :type: numpy.ndarray
    """

    assert len(clusters) == Z.ndim

    labels = [np.unique(c) for c in clusters]

    if nclusters is None:
        nclusters = [None for _ in labels]
    else:
        nclusters = [ncl for ncl in nclusters]

    assert len(nclusters) == Z.ndim

    nclusters = [ncl if ncl is not None else len(lab)
                 for ncl, lab in zip(nclusters, labels)]

    # sort dimensions from largest to smallest
    sorted_dims = np.argsort(nclusters)[::-1]
    features = _calculate_feature(
        Z,
        function,
        labels,
        clusters,
        nclusters,
        sorted_dims,
        **kwargs
    )
    # need to reorder dimensions
    return features.transpose(np.argsort(sorted_dims))


def _calculate_feature(Z, function, labels, clusters, nclusters, axis_order,
                       **kwargs):
    features = np.full([nclusters[ax] for ax in axis_order], np.nan)
    ax, *axis = axis_order
    for label in labels[ax]:
        idx, = np.where(clusters[ax] == label)
        _Z = np.take(Z, idx, axis=ax)
        if axis:
            features[label, ...] = _calculate_feature(
                _Z,
                function,
                labels,
                clusters,
                nclusters,
                axis,
                **kwargs
            )
        else:
            features[label] = function(_Z, **kwargs)
    return features
