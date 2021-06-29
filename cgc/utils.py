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
    C = _est_arr_size((n_cols, nclusters_col), 4)
    R = _est_arr_size((n_rows, nclusters_row), 4)

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
    Calculate the co-cluster averages from the data matrix and the row-
    and column-cluster assignment arrays.

    :param Z: data matrix (m x n)
    :param row_clusters: row clusters (array with size m)
    :param col_clusters: column clusters (array with size n)
    :param nclusters_row: number of row clusters. If not provided,
        determined from the number of unique elements in row_clusters
    :param nclusters_col: number of column clusters. If not provided,
        determined from the number of unique elements in col_clusters
    :return: cocluster averages (nclusters_row, nclusters_col). Co-clusters
        which are not populated are assigned NaN values.
    """
    row_clusters = np.array(row_clusters)
    col_clusters = np.array(col_clusters)

    _row_clusters = np.unique(row_clusters)
    _col_clusters = np.unique(col_clusters)

    nclusters_row = len(_row_clusters) if nclusters_row is None \
        else nclusters_row
    nclusters_col = len(_col_clusters) if nclusters_col is None \
        else nclusters_col

    avg = np.full((nclusters_row, nclusters_col), np.nan)

    # Loop over co-clusters
    for r in _row_clusters:
        idx_rows, = np.where(row_clusters == r)
        for c in _col_clusters:
            idx_cols, = np.where(col_clusters == c)
            ir, ic = np.meshgrid(idx_rows, idx_cols)
            avg[r, c] = Z[ir, ic].mean()
    return avg
