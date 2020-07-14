import concurrent.futures
import dask.distributed

from concurrent.futures import ThreadPoolExecutor
from dask.distributed import Client

from . import coclustering_dask
from . import coclustering_numpy


class Coclustering(object):

    def __init__(self, Z, nclusters_row, nclusters_col, conv_threshold=1.e-5,
                 max_iterations=1, nruns=1, epsilon=1.e-8):
        """

        :param Z:
        :param nclusters_row:
        :param nclusters_col:
        :param conv_threshold:
        :param niterations:
        :param nruns:
        :param epsilon:
        """
        self.Z = Z
        self.nclusters_row = nclusters_row
        self.nclusters_col = nclusters_col
        self.conv_threshold = conv_threshold
        self.max_iterations = max_iterations
        self.nruns = nruns
        self.epsilon = epsilon

        self.client = None

        self.row_clusters = None
        self.col_clusters = None
        self.error = None

    def run_with_dask(self, client=None, low_memory=False):
        """

        :param client: Dask client
        :param low_memory: if true, use a memory-conservative algorithm
        :return:
        """
        self.client = client if client is not None else Client()

        if low_memory:
            self._dask_runs_memory()
        else:
            self._dask_runs_performance()

    def run_with_threads(self, nthreads=1):
        """

        :param nthreads:
        :return:
        """
        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            futures = {
                executor.submit(coclustering_numpy.coclustering,
                                self.Z,
                                self.nclusters_row,
                                self.nclusters_col,
                                self.conv_threshold,
                                self.max_iterations,
                                self.epsilon):
                r for r in range(self.nruns)
            }
            row_min, col_min, e_min = None, None, 0.
            for future in concurrent.futures.as_completed(futures):
                row, col, e = future.result()
                if e < e_min:
                    row_min, col_min, e_min = row, col, e
        self.row_clusters = row_min.compute()
        self.col_clusters = col_min.compute()
        self.error = e_min

    def run_serial(self):
        raise NotImplementedError

    def _dask_runs_memory(self):
        """
        Memory efficient: find minimum-e-run one by one slower because it is
        blocking after each run.
        """
        self.client.scatter(self.Z)
        row_min, col_min, e_min = None, None, 0.
        for r in range(self.nruns):
            print(f'run {r}')
            converged, row, col, e = coclustering_dask.coclustering(
                self.Z,
                self.nclusters_row,
                self.nclusters_col,
                self.conv_threshold,
                self.max_iterations,
                self.epsilon
            )
            if not converged:
                print('WARNING! Not converged')
            if e < e_min:
                row_min, col_min, e_min = row, col, e
        self.row_clusters = row_min.compute()
        self.col_clusters = col_min.compute()
        self.error = e_min

    def _dask_runs_performance(self):
        """
        Performance: find minimum-e-run from all results  faster because there is
        no blocking after each run.
        """
        self.client.scatter(Z)
        futures = [self.client.submit(coclustering_dask.coclustering,
                                      self.Z,
                                      self.nclusters_row,
                                      self.nclusters_col,
                                      self.conv_threshold,
                                      self.max_iterations,
                                      self.epsilon)
                   for r in range(self.nruns)]
        row_min, col_min, e_min = None, None, 0.
        for future, result in dask.distributed.as_completed(futures,
                                                            with_results=True,
                                                            raise_errors=False):
            converged, row, col, e = result
            if not converged:
                print('WARNING! Not converged')
            if e < e_min:
                row_min, col_min, e_min = row, col, e
        self.row_clusters = row_min.compute()
        self.col_clusters = col_min.compute()
        self.error = e_min

