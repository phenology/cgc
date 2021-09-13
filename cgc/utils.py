import math
import logging
import numpy as np


def mem_estimate_coclustering_numpy(n_rows,
                                    n_cols,
                                    nclusters_row,
                                    nclusters_col,
                                    out_unit=None):
    """
    Estimate maximum memory usage of cgc.coclustering_numpy, given the matrix
    size(n_rows, n_cols) and number of row/column clusters (nclusters_row,
    nclusters_col)

    The estimated memory usage is the sum of all major arrays in the process of
    cgc.coclustering_numpy.coclustering.

    Depending on the shape of Z, there are two possible memory peaks: the first
    or the second call of cgc.coclustering_numpy._distance()

    :param n_rows: Number of rows of matrix Z
    :param n_cols: Number of columns of matrix Z
    :param nclusters_row: Number of row clusters
    :param nclusters_col: Number of column clusters
    :param out_unit: Output unit, defaults to None
    :return: Estimated memory usage, unit, peak
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

    return (mem_usage, unit, peak)


def _est_arr_size(shape, nbytes=8):
    """
    Estimate size of an n-D Numpy array

    :param shape: shape of the n-D Numpy array
    :type shape: tuple or list
    :param nbytes: size in bytes of a single element, defaults to 8 (float64)
    :return: Size of the array in bytes
    """

    return np.prod(shape) * nbytes


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
    Calculate the co-cluster averages from the 2D data array and the row-
    and column-cluster assignment arrays.

    :param Z: data array (m x n)
    :param row_clusters: row clusters (array with size m)
    :param col_clusters: column clusters (array with size n)
    :param nclusters_row: number of row clusters. If not provided,
        it is set as the number of unique elements in row_clusters
    :param nclusters_col: number of column clusters. If not provided,
        it is set as the number of unique elements in col_clusters
    :return: co-cluster averages (nclusters_row, nclusters_col). Co-clusters
        that are not populated are assigned NaN values.
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
    Calculate the tri-cluster averages from the 3D data array and the band-,
    row- and column-cluster assignment arrays.

    :param Z: data array (d x m x n)
    :param row_clusters: row clusters (array with size m)
    :param col_clusters: column clusters (array with size n)
    :param bnd_clusters: band clusters (array with size d)
    :param nclusters_row: number of row clusters. If not provided,
        it is set as the number of unique elements in row_clusters
    :param nclusters_col: number of column clusters. If not provided,
        it is set as the number of unique elements in col_clusters
    :param nclusters_bnd: number of band clusters. If not provided,
        it is set as the number of unique elements in bnd_clusters
    :return: tri-cluster averages (nclusters_bnd, nclusters_row, nclusters_col)
        Tri-clusters that are not populated are assigned NaN values.
    """
    return calculate_cluster_feature(
        Z,
        np.mean,
        (bnd_clusters, row_clusters, col_clusters),
        (nclusters_bnd, nclusters_row, nclusters_col)
    )


def calculate_cluster_feature(Z, function, clusters, nclusters=None, **kwargs):
    """
    Calculate feature over clusters

    :param Z:
    :param function:
    :param clusters:
    :param nclusters:
    :param kwargs:
    :return:
    """

    assert len(clusters) == Z.ndim

    labels = [np.unique(c) for c in clusters]

    if nclusters is None:
        nclusters = [len(label) for label in labels]
    assert len(nclusters) == Z.ndim

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
    return features.transpose(sorted_dims)


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
