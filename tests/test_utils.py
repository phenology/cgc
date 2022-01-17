import dask.array as da
import numpy as np

import pytest
from cgc import utils


@pytest.fixture
def test_size_list():
    mat_size_list = [(300000, 50), (500000000, 50), (100, 600000000),
                     (5000, 5000)]
    cls_num_list = [(100, 5), (250, 5), (10, 300), (40, 40)]
    expect_res_list = [(600.85, 'MB', 1), (2165.32, 'GB', 1),
                       (3296.88, 'GB', 2), (195.69, 'MB', 1)]
    test_size = []
    for nrowcol, nrowcol_cls, expect_res in zip(mat_size_list, cls_num_list,
                                                expect_res_list):
        test_size.append([(nrowcol[0], nrowcol[1], nrowcol_cls[0],
                           nrowcol_cls[1]), expect_res])
    return test_size


class TestMemoryEstimation:
    def test_human_size(self):
        input_list = [1024, 1024**2, 1024**3, 1024**4]
        expect_list = [(1., 'KB'), (1., 'MB'), (1., 'GB'), (1024., 'GB')]
        for input, expect in zip(input_list, expect_list):
            assert utils._human_size(input) == expect

    def test_mem_estimate_coclustering_numpy(self, test_size_list):
        for size_input, size_expect in test_size_list:
            size_estimate = utils.mem_estimate_coclustering_numpy(*size_input)
            assert np.abs(size_estimate[0] - size_expect[0]) < 1.
            assert size_estimate[1] == size_expect[1]
            assert size_estimate[2] == size_expect[2]


class TestCalculateCoclusterAverages:
    Z = np.array([
        [0., 0., 1., 1., 2.],
        [1., 1., 2., 2., 3.],
        [1., 1., 2., 2., 3.],
    ])
    row_clusters = np.array([0, 1, 1])
    col_clusters = np.array([2, 2, 0, 0, 1])
    expected_means = np.array([
        [1., 2., 0.],
        [2., 3., 1.]
    ])

    def test_without_specifying_number_of_coclusters(self):
        means = utils.calculate_cocluster_averages(self.Z,
                                                   self.row_clusters,
                                                   self.col_clusters)
        assert means.shape == (2, 3)
        assert np.all(np.isclose(means, self.expected_means))

    def test_without_specifying_number_of_coclusters_dask(self):
        Z = da.from_array(self.Z, chunks=(3, 3))
        means = utils.calculate_cocluster_averages(Z,
                                                   self.row_clusters,
                                                   self.col_clusters)
        assert means.shape == (2, 3)
        assert np.all(np.isclose(means, self.expected_means))

    def test_with_explicit_number_of_clusters(self):
        nclusters_row = len(np.unique(self.row_clusters))
        nclusters_col = len(np.unique(self.col_clusters))
        means = utils.calculate_cocluster_averages(self.Z,
                                                   self.row_clusters,
                                                   self.col_clusters,
                                                   nclusters_row=nclusters_row,
                                                   nclusters_col=nclusters_col)
        assert means.shape == (nclusters_row, nclusters_col)
        assert np.all(np.isclose(means, self.expected_means))

    def test_with_unpopulated_clusters(self):
        nclusters_row = len(np.unique(self.row_clusters))
        # add one column cluster (not populated)
        nclusters_col = len(np.unique(self.col_clusters)) + 1
        means = utils.calculate_cocluster_averages(self.Z,
                                                   self.row_clusters,
                                                   self.col_clusters,
                                                   nclusters_row=nclusters_row,
                                                   nclusters_col=nclusters_col)
        assert means.shape == (nclusters_row, nclusters_col)
        # last column cluster should be full of nan's
        assert np.all(
            np.isclose(
                means[:, -1],
                np.full(nclusters_row, np.nan),
                equal_nan=True
            )
        )
        # the rest of the matrix should be as expected
        assert np.all(
            np.isclose(
                means[:, :-1],
                self.expected_means
            )
        )


class TestCalculateTriclusterAverages:
    Z = np.array([
        [[0., 0., 1., 1., 2.],
         [1., 1., 2., 2., 3.],
         [1., 1., 2., 2., 3.]],
        [[0., 0., 1., 1., 2.],
         [1., 1., 2., 2., 3.],
         [1., 1., 2., 2., 3.]],
        [[4., 4., 5., 5., 6.],
         [5., 5., 6., 6., 7.],
         [5., 5., 6., 6., 7.]],
    ])
    row_clusters = np.array([0, 1, 1])
    col_clusters = np.array([2, 2, 0, 0, 1])
    bnd_clusters = np.array([0, 0, 1])
    expected_means = np.array([
        [[1., 2., 0.],
         [2., 3., 1.]],
        [[5., 6., 4.],
         [6., 7., 5.]],
    ])

    def test_without_specifying_number_of_triclusters(self):
        means = utils.calculate_tricluster_averages(self.Z,
                                                    self.row_clusters,
                                                    self.col_clusters,
                                                    self.bnd_clusters)
        assert means.shape == (2, 2, 3)
        assert np.all(np.isclose(means, self.expected_means))

    def test_without_specifying_number_of_triclusters_dask(self):
        Z = da.from_array(self.Z, chunks=(1, 3, 3))
        means = utils.calculate_tricluster_averages(Z,
                                                    self.row_clusters,
                                                    self.col_clusters,
                                                    self.bnd_clusters)
        assert means.shape == (2, 2, 3)
        assert np.all(np.isclose(means, self.expected_means))

    def test_with_explicit_number_of_triclusters(self):
        nclusters_row = len(np.unique(self.row_clusters))
        nclusters_col = len(np.unique(self.col_clusters))
        nclusters_bnd = len(np.unique(self.bnd_clusters))
        means = utils.calculate_tricluster_averages(
            self.Z,
            self.row_clusters,
            self.col_clusters,
            self.bnd_clusters,
            nclusters_row=nclusters_row,
            nclusters_col=nclusters_col,
            nclusters_bnd=nclusters_bnd
        )
        assert means.shape == (nclusters_bnd, nclusters_row, nclusters_col)
        assert np.all(np.isclose(means, self.expected_means))

    def test_with_unpopulated_clusters(self):
        nclusters_row = len(np.unique(self.row_clusters))
        # add one column cluster (not populated)
        nclusters_col = len(np.unique(self.col_clusters)) + 1
        nclusters_bnd = len(np.unique(self.bnd_clusters))
        means = utils.calculate_tricluster_averages(
            self.Z,
            self.row_clusters,
            self.col_clusters,
            self.bnd_clusters,
            nclusters_row=nclusters_row,
            nclusters_col=nclusters_col,
            nclusters_bnd=nclusters_bnd
        )
        assert means.shape == (nclusters_bnd, nclusters_row, nclusters_col)
        # last column cluster should be full of nan's
        assert np.all(
            np.isclose(
                means[:, :, -1],
                np.full((nclusters_bnd, nclusters_row), np.nan),
                equal_nan=True
            )
        )
        # the rest of the matrix should be as expected
        assert np.all(
            np.isclose(
                means[:, :, :-1],
                self.expected_means
            )
        )


class TestCalculateClusterFeature:
    Z = np.array([
        [0., 0., 1., 1., 2.],
        [1., 1., 2., 2., 3.],
        [1., 1., 2., 2., 3.],
    ])
    row_clusters = np.array([0, 1, 1])
    col_clusters = np.array([2, 2, 0, 0, 1])

    def test_function_without_args(self):
        # all clusters have same values, zero std
        std = utils.calculate_cluster_feature(
            self.Z,
            np.std,
            (self.row_clusters, self.col_clusters)
        )
        assert std.shape == (2, 3)
        assert np.all(np.isclose(std, 0.))

    def test_function_without_args_without_number_of_coclusters(self):
        # all clusters have same values, zero std
        std = utils.calculate_cluster_feature(
            self.Z,
            np.std,
            (self.row_clusters, self.col_clusters)
        )
        assert std.shape == (2, 3)
        assert np.all(np.isclose(std, 0.))

    def test_with_explicit_number_of_clusters(self):
        nclusters_row = len(np.unique(self.row_clusters))
        nclusters_col = len(np.unique(self.col_clusters))
        std = utils.calculate_cluster_feature(
            self.Z,
            np.std,
            (self.row_clusters, self.col_clusters),
            nclusters=(nclusters_row, nclusters_col)
        )
        assert std.shape == (nclusters_row, nclusters_col)
        assert np.all(np.isclose(std, 0.))

    def test_with_transposed_data_matrix(self):
        nclusters_row = len(np.unique(self.row_clusters))
        nclusters_col = len(np.unique(self.col_clusters))
        std = utils.calculate_cluster_feature(
            self.Z.T,
            np.std,
            (self.col_clusters, self.row_clusters),
            nclusters=(nclusters_col, nclusters_row)
        )
        assert std.shape == (nclusters_col, nclusters_row)
        assert np.all(np.isclose(std, 0.))

    def test_function_only_one_number_of_coclusters_is_explicit(self):
        nclusters_row = len(np.unique(self.row_clusters))
        # all clusters have same values, zero std
        std = utils.calculate_cluster_feature(
            self.Z,
            np.std,
            (self.row_clusters, self.col_clusters),
            nclusters=(nclusters_row, None)
        )
        assert std.shape == (2, 3)
        assert np.all(np.isclose(std, 0.))

    def test_with_unpopulated_clusters(self):
        nclusters_row = len(np.unique(self.row_clusters))
        # add one column cluster (not populated)
        nclusters_col = len(np.unique(self.col_clusters)) + 1
        std = utils.calculate_cluster_feature(
            self.Z,
            np.std,
            (self.row_clusters, self.col_clusters),
            nclusters=(nclusters_row, nclusters_col)
        )
        assert std.shape == (nclusters_row, nclusters_col)
        # last column cluster should be full of nan's
        assert np.all(
            np.isclose(
                std[:, -1],
                np.full(nclusters_row, np.nan),
                equal_nan=True
            )
        )
        # the rest of the matrix should be as expected
        assert np.all(np.isclose(std[:, :-1], 0.))
