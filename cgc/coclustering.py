import concurrent.futures
import dask.distributed
import logging

from concurrent.futures import ThreadPoolExecutor
from dask.distributed import Client

from . import coclustering_dask
from . import coclustering_numpy

from .results import Results

logger = logging.getLogger(__name__)


class CoclusteringResults(Results):
    """
    Contains results and metadata of a co-clustering calculation
    """
    def reset(self):
        self.row_clusters_initial = None
        self.col_clusters_initial = None

        self.row_clusters = None
        self.col_clusters = None

        self.error = None
        self.nruns_completed = 0
        self.nruns_converged = 0


class Coclustering(object):
    """
    Perform the co-clustering analysis of a 2D array
    """
    def __init__(self,
                 Z,
                 nclusters_row,
                 nclusters_col,
                 conv_threshold=1.e-5,
                 max_iterations=1,
                 nruns=1,
                 epsilon=1.e-8,
                 low_memory=False,
                 numba_jit=False,
                 output_filename=''):
        """
        Initialize the object

        :param Z: m x n data matrix
        :param nclusters_row: number of row clusters
        :param nclusters_col: number of column clusters
        :param conv_threshold: convergence threshold for the objective function
        :param max_iterations: maximum number of iterations
        :param nruns: number of differently-initialized runs
        :param epsilon: numerical parameter, avoids zero arguments in log
        :param low_memory: boolean parameter, choose low memory implementations
        :param numba_jit: boolean parameter, choose numba optimized single node
        :param output_filename: name of the file where to write the clusters
        """
        self.Z = Z
        self.nclusters_row = nclusters_row
        self.nclusters_col = nclusters_col
        self.conv_threshold = conv_threshold
        self.max_iterations = max_iterations
        self.nruns = nruns
        self.epsilon = epsilon
        self.output_filename = output_filename
        self.client = None

        self.results = CoclusteringResults()

    def run_with_dask(self,
                      client=None,
                      low_memory=False,
                      row_clusters=None,
                      col_clusters=None):
        """
        Run the co-clustering with Dask

        :param client: Dask client
        :param low_memory: if true, use a memory-conservative algorithm
        :param row_clusters: initial row clusters
        :param col_clusters: initial column clusters
        """
        self.client = client if client is not None else Client()

        if row_clusters is not None and col_clusters is not None:
            assert self.nruns == 1, 'Only nruns = 1 for given initial clusters'
            self.results.reset()

        self.results.row_clusters_initial = row_clusters
        self.results.col_clusters_initial = col_clusters

        if low_memory:
            self._dask_runs_memory(row_clusters=row_clusters,
                                   col_clusters=col_clusters)
        else:
            self._dask_runs_performance(row_clusters=row_clusters,
                                        col_clusters=col_clusters)

        self.results.write(filename=self.output_filename)
        return self.results

    def run_with_threads(self,
                         nthreads=1,
                         low_memory=False,
                         numba_jit=False,
                         row_clusters=None,
                         col_clusters=None):
        """
        Run the co-clustering using an algorithm based on numpy + threading
        (only suitable for local runs)

        :param nthreads: number of threads
        :param row_clusters: initial row clusters
        :param col_clusters: initial column clusters
        """
        if row_clusters is not None and col_clusters is not None:
            assert self.nruns == 1, 'Only nruns = 1 for given initial clusters'
            self.results.reset()

        self.results.row_clusters_initial = row_clusters
        self.results.col_clusters_initial = col_clusters

        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            futures = {
                executor.submit(coclustering_numpy.coclustering,
                                self.Z,
                                self.nclusters_row,
                                self.nclusters_col,
                                self.conv_threshold,
                                self.max_iterations,
                                self.epsilon,
                                low_memory,
                                numba_jit,
                                row_clusters_init=row_clusters,
                                col_clusters_init=col_clusters): r
                for r in range(self.nruns)
            }
            for future in concurrent.futures.as_completed(futures):
                logger.info(f'Retrieving run {self.results.nruns_completed}')
                converged, niters, row, col, e = future.result()
                logger.info(f'Error = {e}')
                if converged:
                    logger.info(f'Run converged in {niters} iterations')
                    self.results.nruns_converged += 1
                else:
                    logger.warning(f'Run not converged in {niters} iterations')
                if self.results.error is None or e < self.results.error:
                    self.results.row_clusters = row.tolist()
                    self.results.col_clusters = col.tolist()
                    self.results.error = e
                self.results.nruns_completed += 1
        self.results.write(filename=self.output_filename)
        return self.results

    def _dask_runs_memory(self, row_clusters=None, col_clusters=None):
        """ Memory efficient Dask implementation: sequential runs """
        for r in range(self.nruns):
            logger.info(f'Run {self.results.nruns_completed}')
            converged, niters, row, col, e = coclustering_dask.coclustering(
                self.Z,
                self.nclusters_row,
                self.nclusters_col,
                self.conv_threshold,
                self.max_iterations,
                self.epsilon,
                row_clusters_init=row_clusters,
                col_clusters_init=col_clusters)
            e = e.compute()
            logger.info(f'Error = {e}')
            if converged:
                logger.info(f'Run converged in {niters} iterations')
                self.results.nruns_converged += 1
            else:
                logger.warning(f'Run not converged in {niters} iterations')
            if self.results.error is None or e < self.results.error:
                self.results.row_clusters = row.compute().tolist()
                self.results.col_clusters = col.compute().tolist()
                self.results.error = e
            self.results.nruns_completed += 1

    def _dask_runs_performance(self, row_clusters=None, col_clusters=None):
        """
        Faster but memory-intensive Dask implementation: all runs are
        simultaneosly submitted to the scheduler
        """
        Z = self.client.scatter(self.Z)
        futures = [
            self.client.submit(coclustering_dask.coclustering,
                               Z,
                               self.nclusters_row,
                               self.nclusters_col,
                               self.conv_threshold,
                               self.max_iterations,
                               self.epsilon,
                               row_clusters_init=row_clusters,
                               col_clusters_init=col_clusters,
                               run_on_worker=True,
                               pure=False) for r in range(self.nruns)
        ]
        row_min, col_min, e_min = None, None, 0.
        r = 0
        for future, result in dask.distributed.as_completed(
                futures, with_results=True, raise_errors=False):
            logger.info(f'Waiting for run {r} ..')

            converged, niters, row, col, e = result
            logger.info(f'Error = {e}')
            if converged:
                logger.info(f'Run converged in {niters} iterations')
                self.results.nruns_converged += 1
            else:
                logger.warning(f'Run not converged in {niters} iterations')
            if self.results.error is None or e < self.results.error:
                self.results.row_clusters = row.compute().tolist()
                self.results.col_clusters = col.compute().tolist()
                self.results.error = e
            self.results.nruns_completed += 1
