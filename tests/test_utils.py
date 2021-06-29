import pytest
from cgc import utils


@pytest.fixture
def test_size_list():
    mat_size_list = [(300000, 50), (500000000, 50), (100, 600000000),
                     (5000, 5000)]
    cls_num_list = [(100, 5), (250, 5), (10, 300), (40, 40)]
    expect_res_list = [(686.7, 'MB', 1), (2514.6, 'GB', 1), (3799.8, 'GB', 2),
                       (196.8, 'MB', 1)]
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
            assert size_estimate[0] - size_expect[0] < 1.
            assert size_estimate[1] == size_expect[1]
            assert size_estimate[2] == size_expect[2]
