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
                 output_filename='',
                 row_clusters_init=None,
                 col_clusters_init=None):
        """
        Initialize the object

        :param Z: m x n data matrix
        :param nclusters_row: number of row clusters
        :param nclusters_col: number of column clusters
        :param conv_threshold: convergence threshold for the objective function
        :param max_iterations: maximum number of iterations
        :param nruns: number of differently-initialized runs
        :param epsilon: numerical parameter, avoids zero arguments in log
        :param output_filename: name of the file where to write the clusters
        :param row_clusters_init: initial row clusters
        :param col_clusters_init: initial column clusters
        """
        # Input parameters -----------------
        self.nclusters_row = nclusters_row
        self.nclusters_col = nclusters_col
        self.conv_threshold = conv_threshold
        self.max_iterations = max_iterations
        self.nruns = nruns
        self.epsilon = epsilon
        self.output_filename = output_filename
        self.row_clusters_init = row_clusters_init
        self.col_clusters_init = col_clusters_init
        # Input parameters end -------------

        # store input parameters in results object
        self.results = CoclusteringResults(**self.__dict__)

        assert Z.ndim == 2, 'Incorrect dimensionality for Z matrix'
        self.Z = Z

        if row_clusters_init is not None and col_clusters_init is not None:
            assert nruns == 1, 'Only nruns = 1 for given initial clusters'
            assert Z.shape == (len(row_clusters_init), len(col_clusters_init))

        self.client = None

    def run_with_dask(self, client=None, low_memory=False):
        """
        Run the co-clustering with Dask

        :param client: Dask client
        :param low_memory: if true, use a memory-conservative algorithm
        :return: co-clustering results
        """
        self.client = client if client is not None else Client()

        if low_memory:
            self._dask_runs_memory()
        else:
            self._dask_runs_performance()

        self.results.write(filename=self.output_filename)
        return self.results

    def run_with_threads(self,
                         nthreads=1,
                         low_memory=False,
                         numba_jit=False):
        """
        Run the co-clustering using an algorithm based on numpy + threading
        (only suitable for local runs)

        :param nthreads: number of threads
        :param low_memory: if true, use a memory-conservative algorithm
        :param numba_jit: if true, and low_memory is true, then use Numba
                          just-in-time compilation to improve performance
        :return: co-clustering results
        """
        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            futures = [
                executor.submit(coclustering_numpy.coclustering,
                                self.Z,
                                self.nclusters_row,
                                self.nclusters_col,
                                self.conv_threshold,
                                self.max_iterations,
                                self.epsilon,
                                low_memory,
                                numba_jit,
                                row_clusters_init=self.row_clusters_init,
                                col_clusters_init=self.col_clusters_init)
                for _ in range(self.nruns)
            ]
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
                    self.results.row_clusters = row
                    self.results.col_clusters = col
                    self.results.error = e
                self.results.nruns_completed += 1
        self.results.write(filename=self.output_filename)
        return self.results

    def _dask_runs_memory(self):
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
                row_clusters_init=self.row_clusters_init,
                col_clusters_init=self.col_clusters_init
            )
            logger.info(f'Error = {e}')
            if converged:
                logger.info(f'Run converged in {niters} iterations')
                self.results.nruns_converged += 1
            else:
                logger.warning(f'Run not converged in {niters} iterations')
            if self.results.error is None or e < self.results.error:
                self.results.row_clusters = row.compute()
                self.results.col_clusters = col.compute()
                self.results.error = e
            self.results.nruns_completed += 1

    def _dask_runs_performance(self):
        """
        Faster but memory-intensive Dask implementation: all runs are
        simultaneosly submitted to the scheduler
        """
        Z = self.client.scatter(self.Z)
        futures = [self.client.submit(
                       coclustering_dask.coclustering,
                       Z,
                       self.nclusters_row,
                       self.nclusters_col,
                       self.conv_threshold,
                       self.max_iterations,
                       self.epsilon,
                       row_clusters_init=self.row_clusters_init,
                       col_clusters_init=self.col_clusters_init,
                       run_on_worker=True,
                       pure=False)
                   for _ in range(self.nruns)]
        for future, result in dask.distributed.as_completed(futures,
                                                            with_results=True):
            logger.info(f'Retrieving run {self.results.nruns_completed} ..')
            converged, niters, row, col, e = result
            logger.info(f'Error = {e}')
            if converged:
                logger.info(f'Run converged in {niters} iterations')
                self.results.nruns_converged += 1
            else:
                logger.warning(f'Run not converged in {niters} iterations')
            if self.results.error is None or e < self.results.error:
                self.results.row_clusters = row.compute()
                self.results.col_clusters = col.compute()
                self.results.error = e
            self.results.nruns_completed += 1
