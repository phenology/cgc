from osgeo import gdal
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def binarize(M):
    uniq = np.unique(M)
    YY = np.zeros((M.shape[0], len(uniq)))
    for i in range(0, len(uniq)):
        YY[np.where(M == uniq[i])[0], i] = 1
    return YY


def scale(X):
    cols = []
    descale = []
    for feature in X.T:
        minimum = np.nanmin(feature, axis=0)
        maximum = np.nanmax(feature, axis=0)
        col_std = np.divide((feature - minimum), (maximum - minimum))
        cols.append(col_std)
        descale.append((minimum, maximum))
    X_std = np.array(cols)
    return X_std.T, descale


# # Path of co-clustering results:
root = '/media/emma/emma/eScience/results/'
# # Output path to save the results:
rootOutput = '/media/emma/emma/eScience/results/'
# Variables of the co-clustering:
type = 'Bloom'
gr_row = '70'
gr_col = '4'
clus = '15'
iter = '100'

file = type + 'Final_one_band_' + gr_row + '_' + gr_col + '_' + clus + '_' + iter + '_new.tif'
root_image_cc = root + file
file_csv = type + 'Final_one_band_' + gr_row + '_' + gr_col + '_' + clus + '_' + iter + '_new.csv'
root_csv = root + file_csv

# # We have to load the temporal series data:
years = np.linspace(1990, 2014, 25)
# Path:
folder = '/media/emma/emma/eScience/Alliance1/Data/' + type + 'Trocitos/'

# Load the first image of temporal series:
root_image = folder + type + '_1989.tif'
tifsrc = gdal.Open(root_image)

in_band = tifsrc.GetRasterBand(1)
block_xsize = in_band.XSize
block_ysize = in_band.YSize

# read the multiband tile into a 3d numpy array
matrix = tifsrc.ReadAsArray(0, 0, block_xsize, block_ysize)
matrix = np.reshape(matrix, (matrix.shape[0] * matrix.shape[1], 1))

# Load the rest of images of temporal series:
for yr in years:
    root_image = folder + type + '_' + str(yr)[0:4] + '.tif'
    tifsrc = gdal.Open(root_image)

    in_band = tifsrc.GetRasterBand(1)
    block_xsize = in_band.XSize
    block_ysize = in_band.YSize

    # read the multiband tile into a 3d numpy array
    image = tifsrc.ReadAsArray(0, 0, block_xsize, block_ysize)
    image = np.reshape(image, (image.shape[0] * image.shape[1], 1))
    matrix = np.concatenate((matrix, image), axis=1)

# Load the geotiff obtained in co-clustering:
tifsrc = gdal.Open(root_image_cc)

in_band = tifsrc.GetRasterBand(1)
block_xsize = in_band.XSize
block_ysize = in_band.YSize

# read the multiband tile into a 3d numpy array
R = tifsrc.ReadAsArray(0, 0, block_xsize, block_ysize)
R = np.reshape(R, (R.shape[0]*R.shape[1], 1))

# # Load the csv file:
C = pd.read_csv(root_csv, header=0)

# Statistics measures (Mean, STD, 5 percentile, 95 percentile, maximum and minimum values) from each co-cluster group:
ind_gr_row_label = np.where(~np.isnan(np.unique(R)))[0]
gr_rows = np.unique(R)[ind_gr_row_label]
X = np.empty([0, 6])
for r in gr_rows:
    for c in range(0, C.shape[1]):

        temp_cl = np.where(C[C.columns[c]].values)[0]
        spatial_cl = np.where(R == r)
        Index_temp = matrix[spatial_cl[0], :][:, temp_cl]

        data = np.array([np.nanmean(Index_temp), np.nanstd(Index_temp), np.nanpercentile(Index_temp, 5),
                         np.nanpercentile(Index_temp, 95), np.nanmax(Index_temp), np.nanmin(Index_temp)])

        X = np.vstack((X, data))

# # Kmeans:
X = scale(X)
kmeans_cc = KMeans(n_clusters=5, max_iter=500).fit(X[0])

# # Obtain the centroids (mean values) of each group and they are transformed to matrix:
mean_centroids = (kmeans_cc.cluster_centers_[:, 0]*(X[1][0][1]-X[1][0][0]))+X[1][0][0]
ircc_co = mean_centroids[kmeans_cc.labels_]
ircc_co = np.reshape(ircc_co, (int(gr_row), int(gr_col)))

# The centroid matrix is expanded to the size of original data to plot the spatial information within temporal
# groups and the temporal information within spatial groups:
C = C.to_numpy()
ircc = np.dot(ircc_co, C.T)
R_binary = binarize(R)
ircc = np.dot(R_binary, ircc)

# #### Visualization part ######

# ############ Spatial Grid  #############
fig, ax = plt.subplots(int(np.floor(C.shape[1]/2)), int(C.shape[1]-np.floor(C.shape[1]/2)))
empty_string_labels = ['']
min_val = np.min(ircc)
max_val = np.max(ircc)
colorbar_th = np.round(np.linspace(np.ceil(min_val), np.floor(max_val), 10))
for c, a in zip(range(0, C.shape[1]), ax.flatten()):
    temp_cl = np.where(C[:, c])[0]
    pr = np.unique(ircc[:, temp_cl], axis=1)
    spatial_group = np.reshape(pr, (block_ysize, block_xsize))
    fig1 = a.imshow(spatial_group, interpolation="None", vmin=min_val, vmax=max_val)
    a.set_title('Temp. cl ' + str(c+1), fontsize=16)
    a.grid(True)
    a.set_xticklabels(empty_string_labels)
    a.set_yticklabels(empty_string_labels)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
cb = fig.colorbar(fig1, cax=cbar_ax, ticks=colorbar_th)
cb.set_label('DOY', fontsize=16)
cb.ax.tick_params(labelsize=18)
plt.savefig(rootOutput + 'Spatial_info.png', format='png', transparent=True, bbox_inches="tight")
plt.close(fig)

# ############ Temporal plot #############
S = []
for i in np.unique(kmeans_cc.labels_):
    S.append(str(i+1))

plt.plot(years, np.where(C)[1], marker='o', color="blue", alpha=0.4)
plt.xlabel('Year', fontsize=20)
plt.ylabel('Spatial cluster', fontsize=20)
plt.tick_params(labelsize=16)
plt.grid(True)
plt.yticks(np.arange(len(S)-1), np.array(S), color='k', size=20)
plt.savefig(rootOutput + 'Temporal_info.png', format='png', transparent=True, bbox_inches="tight")
