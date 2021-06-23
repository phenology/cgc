import math
import logging
import sys

logging.basicConfig(level=logging.INFO,
                    stream=sys.stdout,
                    format="%(asctime)-15s %(levelname)-5s - %(message)s")


def mem_estimate_numpy(m, n, k, l, out_unit=None):
    """
    Estimate maximum memory usage of cgc.coclustering_numpy, given the matrix 
    size(m, n) and number of row/column clusters (k, l)

    The estimated memory usage is the sum of all major arrays in the process of
    cgc.coclustering_numpy.coclustering. 
    
    Depending on the shape of Z, there are two possible memory peaks: the first
    or the second call of cgc.coclustering_numpy._distance()

    :param m: Number of rows of matrix Z
    :param n: Number of columns of matrix Z
    :param k: Number of row clusters
    :param l: Number of column clusters
    :param out_unit: Output unit, defaults to None
    :return: Estimated memory usage, unit, peak
    """

    # Size of major matrix
    Z = _est_arr_size((m, n))
    C = _est_arr_size((n, l), 4)
    R = _est_arr_size((m, k), 4)

    # First call _distance
    Y1 = _est_arr_size((n, k))
    d1 = _est_arr_size((m, k)) * 2  # x2 because "Y.sum(axis=0, keepdims=True)"
    # is also exanpded to (m,k) for matrix minus
    # operation

    # Second call _distance
    Y2 = _est_arr_size((m, l))
    d2 = _est_arr_size((n, l)) * 2  # x2 because "Y.sum(axis=0, keepdims=True)"
    # is also exanpded to (m,k) for matrix minus
    # operation

    # Compare two potential peaks
    mem1 = Z + C + R + Y1 + d1
    mem2 = Z + C + R + Y2 + d2
    mem_usage = max([mem1, mem2])
    peak = [mem1, mem2].index(mem_usage) + 1

    # Make results human redable
    mem_usage, unit = _human_size(mem_usage, out_unit)

    return (mem_usage, unit, peak)


def _est_arr_size(shape, nbytes=8):
    """
    Estimate size of an n-D Numpy array

    :param shape: shape of the n-D Numpy array
    :type shape: tuple or list
    :param nbytes: size in bytes of a single element, defaults to 8 (float64)
    :return: Size of the array in bytes
    """

    return math.prod(shape) * nbytes


import math


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
