import logging
import time
import sys
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt

from cgc.kmeans import Kmeans
from cgc.coclustering import Coclustering

if __name__ == "__main__":

    # Co-clustering
    k = 30  # num clusters in rows
    l = 5  # num clusters in columns
    errobj, niters, nruns, epsilon = 0.00001, 1, 1, 10e-8
    Z = np.random.randint(1000, size=(100000, 20))
    Z = Z.astype('float64')
    cc = Coclustering(Z, k, l, errobj, niters, nruns, epsilon)
    cc.run_with_threads(nthreads=1)

    # Kmean
    kmean_max_iter = 500
    km = Kmeans(Z=Z,
                row_clusters=cc.row_clusters,
                col_clusters=cc.col_clusters,
                n_row_clusters=k,
                n_col_clusters=l,
                k_range=range(2, 25),
                kmean_max_iter=kmean_max_iter,
                ouputdir='.')
    km.compute()
    km.cl_mean_centroids
