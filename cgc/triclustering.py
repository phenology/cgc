import copy
import concurrent.futures
import logging

from concurrent.futures import ThreadPoolExecutor
from dask.distributed import Client

from . import triclustering_dask
from . import triclustering_numpy

from .results import Results

logger = logging.getLogger(__name__)


class TriclusteringResults(Results):
    """
    Contains results and metadata of a tri-clustering calculation.

    :var row_clusters: Final row cluster assignment.
    :type row_clusters: numpy.ndarray
    :var col_clusters: Final column cluster assignment.
    :type col_clusters: numpy.ndarray
    :var bnd_clusters: Final band cluster assignment.
    :type bnd_clusters: numpy.ndarray
    :var error: Approximation error of the tri-clustering.
    :type error: float
    :var nruns_completed: Number of successfully completed runs.
    :type nruns_completed: int
    :var nruns_converged: Number of converged runs.
    :type nruns_converged: int
    """
    row_clusters = None
    col_clusters = None
    bnd_clusters = None
    error = None
    nruns_completed = 0
    nruns_converged = 0


class Triclustering(object):
    """
    Perform a tri-clustering analysis for a three-dimensional array.

    :param Z: Data array for which to run the tri-clustering analysis, with
        shape (`band`, `row`, `column`).
    :type Z: numpy.ndarray or dask.array.Array
    :param nclusters_row: Number of row clusters.
    :type nclusters_row: int
    :param nclusters_col: Number of column clusters.
    :type nclusters_col: int
    :param nclusters_bnd: Number of band clusters.
    :type nclusters_bnd: int
    :param conv_threshold: Convergence threshold for the objective function.
    :type conv_threshold: float, optional
    :param max_iterations: Maximum number of iterations.
    :type max_iterations: int, optional
    :param nruns: Number of differently-initialized runs.
    :type nruns: int, optional
    :param output_filename: Name of the JSON file where to write the results.
    :type output_filename: string, optional
    :param row_clusters_init: Initial row cluster assignment.
    :type row_clusters_init: numpy.ndarray or array_like, optional
    :param col_clusters_init: Initial column cluster assignment.
    :type col_clusters_init: numpy.ndarray or array_like, optional
    :param bnd_clusters_init: Initial band cluster assignment.
    :type bnd_clusters_init: numpy.ndarray or array_like, optional

    :Example:

    >>> import numpy as np
    >>> Z = np.random.randint(1, 100, size=(6, 10, 8)).astype('float64')
    >>> tc = Triclustering(Z,
                          nclusters_row=5,
                          nclusters_col=4,
                          max_iterations=50,
                          nruns=10)
    """
    def __init__(self,
                 Z,
                 nclusters_row,
                 nclusters_col,
                 nclusters_bnd,
                 conv_threshold=1.e-5,
                 max_iterations=1,
                 nruns=1,
                 output_filename='',
                 row_clusters_init=None,
                 col_clusters_init=None,
                 bnd_clusters_init=None):
        # Input parameters -----------------
        self.nclusters_row = nclusters_row
        self.nclusters_col = nclusters_col
        self.nclusters_bnd = nclusters_bnd
        self.conv_threshold = conv_threshold
        self.max_iterations = max_iterations
        self.nruns = nruns
        self.output_filename = output_filename
        self.row_clusters_init = row_clusters_init
        self.col_clusters_init = col_clusters_init
        self.bnd_clusters_init = bnd_clusters_init
        # Input parameters end -------------

        # store input parameters in results object
        self.results = TriclusteringResults(**self.__dict__)

        assert Z.ndim == 3, 'Incorrect dimensionality for Z matrix'
        self.Z = Z

        if all([
                cls is not None for cls in
                [row_clusters_init, col_clusters_init, bnd_clusters_init]
        ]):
            assert nruns == 1, 'Only nruns = 1 for given initial clusters'
            assert Z.shape == (len(bnd_clusters_init), len(row_clusters_init),
                               len(col_clusters_init))

        self.client = None

    def run_with_threads(self, nthreads=1):
        """
        Run the tri-clustering using an algorithm based on Numpy plus threading
        (only suitable for local runs).

        :param nthreads: Number of threads employed to simultaneously run
            differently-initialized tri-clustering analysis.
        :type nthreads: int, optional
        :return: tri-clustering results.
        :type: cgc.triclustering.TriclusteringResults
        """
        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            futures = {
                executor.submit(triclustering_numpy.triclustering,
                                self.Z,
                                self.nclusters_row,
                                self.nclusters_col,
                                self.nclusters_bnd,
                                self.conv_threshold,
                                self.max_iterations,
                                row_clusters_init=self.row_clusters_init,
                                col_clusters_init=self.col_clusters_init,
                                bnd_clusters_init=self.bnd_clusters_init): r
                for r in range(self.nruns)
            }
            for future in concurrent.futures.as_completed(futures):
                logger.info(f'Waiting for run {self.results.nruns_completed}')
                converged, niters, row, col, bnd, e = future.result()
                logger.info(f'Error = {e}')
                if converged:
                    logger.info(f'Run converged in {niters} iterations')
                    self.results.nruns_converged += 1
                else:
                    logger.warning(f'Run not converged in {niters} iterations')
                if self.results.error is None or e < self.results.error:
                    self.results.row_clusters = row
                    self.results.col_clusters = col
                    self.results.bnd_clusters = bnd
                    self.results.error = e
                self.results.nruns_completed += 1
        self.results.write(filename=self.output_filename)
        return copy.copy(self.results)

    def run_with_dask(self, client=None):
        """
        Run the tri-clustering analysis using Dask.

        :param client: Dask client. If not specified, the default
            `LocalCluster` is employed.
        :type client: dask.distributed.Client, optional
        :return: Tri-clustering results.
        :type: cgc.triclustering.TriclusteringResults
        """
        self.client = client if client is not None else Client()
        self._dask_runs_memory()
        self.results.write(filename=self.output_filename)
        return copy.copy(self.results)

    def _dask_runs_memory(self):
        """ Memory efficient Dask implementation: sequential runs """
        for r in range(self.nruns):
            logger.info(f'Run {self.results.nruns_completed}')
            converged, niters, row, col, bnd, e = \
                triclustering_dask.triclustering(
                    self.Z,
                    self.nclusters_row,
                    self.nclusters_col,
                    self.nclusters_bnd,
                    self.conv_threshold,
                    self.max_iterations,
                    row_clusters_init=self.row_clusters_init,
                    col_clusters_init=self.col_clusters_init,
                    bnd_clusters_init=self.bnd_clusters_init
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
                self.results.bnd_clusters = bnd.compute()
                self.results.error = e
            self.results.nruns_completed += 1

    def run_serial(self):
        raise NotImplementedError
