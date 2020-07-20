import logging
import time
import sys
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt

from geoclustering.kmeans import Kmeans
from geoclustering.coclustering import Coclustering

if __name__ == "__main__":

    # Co-clustering
    k = 25  # num clusters in rows
    l = 5  # num clusters in columns
    errobj, niters, nruns, epsilon = 0.00001, 1, 1, 10e-8
    Z = np.load('../testdata/LeafFinal_one_band_3000000_1980-2017_int32.npy')
    Z = Z.astype('float64')
    cc = Coclustering(Z, k, l, errobj, niters, nruns, epsilon)
    cc.run_with_threads(nthreads=1)


    # Kmean
    kmean_n_clusters = 5
    kmean_max_iter = 500
    km = Kmeans(Z=Z,
                row_clusters=cc.row_clusters,
                col_clusters=cc.col_clusters,
                n_row_clusters=k,
                n_col_clusters=l,
                kmean_n_clusters=kmean_n_clusters,
                kmean_max_iter=kmean_max_iter)
    km.compute()
    km.cl_mean_centroids


    # Set up visualization
    fig_outputdir = '../testdata/test_kmean_results'
    block_ysize = 1500 # Check
    block_xsize = 2000 # Check
    years = np.linspace(1980, 2017, 38)
    R = np.eye(k)[cc.row_clusters]
    C = np.eye(l)[cc.col_clusters]
    # Remove the nan row/col according to km.cl_mean_centroids
    idx_nan_row = np.all(np.isnan(km.cl_mean_centroids), axis=1)
    idx_nan_col = np.all(np.isnan(km.cl_mean_centroids), axis=0)
    cl_mean_centroids_nonnan = np.delete(km.cl_mean_centroids,
                                         np.where(idx_nan_row)[0],
                                         axis=0)
    cl_mean_centroids_nonnan = np.delete(cl_mean_centroids_nonnan,
                                         np.where(idx_nan_col)[0],
                                         axis=1)
    R = np.delete(R, np.where(idx_nan_row)[0], axis=1)
    C = np.delete(C, np.where(idx_nan_col)[0], axis=1)
    # The centroid matrix is expanded to the size of original data to plot the
    # spatial information within temporal groups and the temporal information
    # within spatial groups:
    ircc = np.dot(np.dot(R, cl_mean_centroids_nonnan), C.T)


    # #Plot Spatial Grid
    # fig, ax = plt.subplots(int(np.floor(C.shape[1]/2)),
    #                        int(C.shape[1]-np.floor(C.shape[1]/2)))
    # empty_string_labels = ['']
    # min_val = np.min(ircc)
    # max_val = np.max(ircc)
    # colorbar_th = np.round(np.linspace(np.ceil(min_val),
    #                                    np.floor(max_val), 10))
    # for c, a in zip(range(0, C.shape[1]), ax.flatten()):
    #     temp_cl = np.where(C[:, c])[0]
    #     pr = np.unique(ircc[:, temp_cl], axis=1)
    #     spatial_group = np.reshape(pr, (block_ysize, block_xsize))
    #     fig1 = a.imshow(spatial_group, interpolation="None", vmin=min_val,
    #                     vmax=max_val)
    #     a.set_title('Temp. cl ' + str(c+1), fontsize=16)
    #     a.grid(True)
    #     a.set_xticklabels(empty_string_labels)
    #     a.set_yticklabels(empty_string_labels)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # cb = fig.colorbar(fig1, cax=cbar_ax, ticks=colorbar_th)
    # cb.set_label('DOY', fontsize=16)
    # cb.ax.tick_params(labelsize=18)
    # plt.savefig(fig_outputdir + 'Spatial_info.png', format='png',
    #             transparent=True, bbox_inches="tight")
    # plt.close(fig)

    # ############ Temporal plot #############
    S = []
    for i in np.unique(km.kmeans_cc.labels_):
        S.append(str(i+1))
    plt.plot(years, np.where(C)[1], marker='o', color="blue", alpha=0.4)
    plt.xlabel('Year', fontsize=20)
    plt.ylabel('Spatial cluster', fontsize=20)
    plt.tick_params(labelsize=16)
    plt.grid(True)
    plt.yticks(np.arange(len(S)-1), np.array(S), color='k', size=20)
    plt.savefig(fig_outputdir + 'Temporal_info.png', format='png',
                transparent=True, bbox_inches="tight")
