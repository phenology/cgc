import logging
import time
import sys
import dask.array as da
import numpy as np

from geoclustering.kmeans import Kmeans
from geoclustering.coclustering import Coclustering

if __name__ == "__main__":
    
    # Co-clustering
    k = 25  # num clusters in rows
    l = 5  # num clusters in columns
    errobj, niters, nruns, epsilon = 0.00001, 1, 1, 10e-8
    Z = np.load('/mnt/c/Users/OuKu/Developments/phenology/data/LeafFinal_one_band_3000000_1980-2017_int32.npy')
    Z = Z.astype('float64')
    cc = Coclustering(Z, k, l, errobj, niters, nruns, epsilon)
    cc.run_with_threads(nthreads=1)
    
    # Kmean
    kmean_n_clusters = 5
    kmean_max_iter = 500
    km = Kmeans(Z = Z, 
                row_clusters = cc.row_clusters, 
                col_clusters = cc.col_clusters, 
                n_row_clusters = k,
                n_col_clusters = l,
                kmean_n_clusters = kmean_n_clusters,
                kmean_max_iter = kmean_max_iter)
    km.compute()
    km.cl_mean_centroids

    
    