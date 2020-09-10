import logging
import time
import sys
import numpy as np
#import matplotlib.pyplot as plt

from cgc.coclustering import Coclustering

if __name__ == "__main__":

    # Co-clustering with numba
    k = 10  # num clusters in rows
    l = 5 # num clusters in columns
    errobj, niters, nruns, epsilon = 0.00001, 300, 1, 10e-8
    #Z = np.load('../testdata/LeafFinal_one_band_3000000_1980-2017_int32.npy')
    #Z = Z.astype('float64')
    m = 100000
    n = 60
    Z = np.random.randint(100, size=(m, n)).astype('float64')
    r_cluster = np.mod(np.arange(m), k)
    c_cluster = np.mod(np.arange(n), l)

    numba_start_time = time.time()
    ccnumba = Coclustering(Z, k, l, errobj, niters, nruns, epsilon,numba_jit=False)
    ccnumba.run_with_threads(nthreads=1, row_clusters = r_cluster, col_clusters=c_cluster, low_memory=True)
    numba_end_time = time.time()
    numba_elapsed_time = numba_end_time - numba_start_time


    # Co-clustering with vanilla numpy
    #Z = np.random.randint(100, size=(20000, 150)).astype('float64')
    numpy_start_time = time.time()
    cc = Coclustering(Z, k, l, errobj, niters, nruns, epsilon)
    cc.run_with_threads(nthreads=1, low_memory=True, row_clusters = r_cluster, col_clusters=c_cluster)
    numpy_end_time = time.time()
    numpy_elapsed_time = numpy_end_time - numpy_start_time

    diff_rowcl = np.array(ccnumba.results.row_clusters) - np.array(cc.results.row_clusters)
    diff_colcl =  np.array(ccnumba.results.col_clusters) - np.array(cc.results.col_clusters)

    print('Coclustering elpased times')
    print('--------------------------')
    print('Settings:')
    print('Matrix MxN :','M', m, 'N', n)
    print('number of clusters k, l:', k, l)
    print('elapsed time')
    print('numba :', numba_elapsed_time)
    print('numpy elaspsed time:', numpy_elapsed_time)
    print('diff row_cluster:', diff_rowcl)
    print('diff col_cluster:', diff_colcl)

    pass