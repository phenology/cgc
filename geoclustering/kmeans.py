import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from osgeo import gdal
from sklearn.cluster import KMeans

import concurrent.futures
import dask.distributed
import logging

from concurrent.futures import ThreadPoolExecutor
from dask.distributed import Client

logger = logging.getLogger(__name__)


class Kmeans(object):

    def __init__(self, Z, row_clusters, col_clusters, n_row_clusters, n_col_clusters, kmean_n_clusters, kmean_max_iter=100, output_dir='.'):
        """

        :param Z: m x n matrix of spatial-temporal data. Usually each row is a time-series of a spatial grid. 
        :param row_clusters: m x 1 row cluster array.
        :param col_clusters: n x 1 column cluster array.
        :param n_row_clusters: number of row clusters
        :param n_col_clusters: number of column clusters
        :param kmean_n_clusters: number of clusters to form in KMean, i.e. value "k"
        :param kmean_max_iter: maximum number of iterations of the KMeans
        :param output_dir: results output directory
        """
        self.Z = Z
        self.row_clusters = row_clusters
        self.col_clusters = col_clusters
        self.n_row_clusters = n_row_clusters
        self.n_col_clusters = n_col_clusters
        self.kmean_n_clusters = kmean_n_clusters
        self.kmean_max_iter = kmean_max_iter
        self.output_dir = output_dir

        if len(np.unique(row_clusters))>n_row_clusters or len(np.unique(col_clusters))>n_col_clusters:
            raise ValueError


    def compute(self):
        self._statistic_mesures()
        self._compute_kmean()


    def _statistic_mesures(self):
        """
        Compute 6 statistics: (Mean, STD, 5 percentile, 95 percentile, maximum and minimum values) from each co-cluster group:
        """
        self.stat_measures = np.empty([0, 6])        
        # Loop per co-cluster cell
        for r in np.unique(self.row_clusters):
            for c in np.unique(self.col_clusters):
                idx_rows = np.where(self.row_clusters == r)[0]
                idx_col = np.where(self.col_clusters == c)[0]
                cl_Z = self.Z[idx_rows, :][:, idx_col] # All elements in Z falliing into this cluster cell

                cl_stat = np.array([np.nanmean(cl_Z), np.nanstd(cl_Z), np.nanpercentile(cl_Z, 5),
                                np.nanpercentile(cl_Z, 95), np.nanmax(cl_Z), np.nanmin(cl_Z)])

                self.stat_measures = np.vstack((self.stat_measures, cl_stat))

    def _compute_kmean(self):
        """
        Compute kmean:
        """
        # Normalize all statistic measures to [0, 1]
        stat_measures_norm = []
        descale = []
        for sm in self.stat_measures.T:
            minimum = np.nanmin(sm, axis=0)
            maximum = np.nanmax(sm, axis=0)
            sm_norm = np.divide((sm - minimum), (maximum - minimum))
            stat_measures_norm.append(sm_norm)
        
        self.stat_measures_norm = np.array(stat_measures_norm).T 

        # Compute Kmean
        self.kmeans_cc = KMeans(n_clusters=self.kmean_n_clusters, max_iter=self.kmean_max_iter).fit(self.stat_measures_norm)

        # Get centroids of the "mean value" dimension, and scale back
        # TODO: do we need centroids of other statistic measures?

        mean_centroids_norm = self.kmeans_cc.cluster_centers_[:, 0]
        max_mean = np.nanmax(self.stat_measures[:,0])
        min_mean = np.nanmin(self.stat_measures[:,0])
        mean_centroids = mean_centroids_norm * (max_mean - min_mean) + min_mean
        
        # Assign centroids to each cluster cell
        cl_mean_centroids = mean_centroids[self.kmeans_cc.labels_]

        # Reshape to the shape of cluster matrix, taking into account non-constructive row/col cluster 
        self.cl_mean_centroids = np.empty((self.n_row_clusters, self.n_col_clusters))
        self.cl_mean_centroids[:] = np.nan
        idx=0
        for r in np.unique(self.row_clusters):
            for c in np.unique(self.col_clusters):
                self.cl_mean_centroids[r,c] = cl_mean_centroids[idx]
                idx = idx+1