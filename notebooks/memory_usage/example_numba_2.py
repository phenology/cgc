import logging
import time
import sys
import numpy as np
import cProfile
import numba

from cgc.coclustering_numpy import _initialize_clusters
from cgc.coclustering_numpy import _cluster_dot
from cgc.coclustering_numpy import _cluster_dot_numba
from cgc.coclustering_numpy import _distance
from cgc.coclustering_numpy import _distance_lowmem_numba
from cgc.coclustering_numpy import _distance_lowmem
from cgc.coclustering_numpy import coclustering

k = 15
l = 10
mm = 100000
nn = 70
Z = np.random.randint(100, size=(mm, nn)).astype('float64')

nclusters_col = k
nclusters_row = l

errobj, niters, nruns, epsilon = 0.00001, 300, 20, 10e-8


def time_coc_n(Z, nclusters_col, nclusters_row, errobj, niters, epsilon):
    st = time.time()
    converged, s, row_clusters, col_clusters, e = coclustering(Z,
                                                               nclusters_col,
                                                               nclusters_row,
                                                               errobj,
                                                               niters,
                                                               epsilon,
                                                               low_memory=True,
                                                               numba_jit=True)
    et = time.time()
    dt = et - st
    return dt, converged, s, row_clusters, col_clusters, e

def time_coc(Z, nclusters_col, nclusters_row, errobj, niters, epsilon):
    st = time.time()
    converged, s, row_clusters, col_clusters, e = coclustering(Z,
                                                               nclusters_col,
                                                               nclusters_row,
                                                               errobj,
                                                               niters,
                                                               epsilon,
                                                               low_memory=False,
                                                               numba_jit=False)
    et = time.time()
    dt = et - st
    return dt, converged, s, row_clusters, col_clusters, e


@numba.jit(nopython=True, nogil=True, parallel=True, cache=True, fastmath=True)
def time_coc_l(Z, nclusters_col, nclusters_row, errobj, niters, epsilon):
    st = time.time()
    converged, s, row_clusters, col_clusters, e = coclustering(Z,
                                                               nclusters_col,
                                                               nclusters_row,
                                                               errobj,
                                                               niters,
                                                               epsilon,
                                                               low_memory=True,
                                                               numba_jit=False)
    et = time.time()
    dt = et - st
    return dt, converged, s, row_clusters, col_clusters, e
