import concurrent.futures
import logging

from concurrent.futures import ThreadPoolExecutor

from . import triclustering_numpy

from .results import Results

logger = logging.getLogger(__name__)


class TriclusteringResults(Results):
    """
    Contains results and metadata of a tri-clustering calculation
    """
    def reset(self):
        self.row_clusters = None
        self.col_clusters = None
        self.bnd_clusters = None
        self.error = None
        self.nruns_completed = 0
        self.nruns_converged = 0


class Triclustering(object):
    """
    Perform the tri-clustering analysis of a 3D array
    """
    def __init__(self, Z, nclusters_row, nclusters_col, nclusters_bnd,
                 conv_threshold=1.e-5, max_iterations=1, nruns=1,
                 epsilon=1.e-8, output_filename='', row_clusters_init=None,
                 col_clusters_init=None, bnd_clusters_init=None):
        """
        Initialize the object

        :param Z: d x m x n data matrix
        :param nclusters_row: number of row clusters
        :param nclusters_col: number of column clusters
        :param nclusters_bnd: number of band clusters
        :param conv_threshold: convergence threshold for the objective function
        :param max_iterations: maximum number of iterations
        :param nruns: number of differntly-initialized runs
        :param epsilon: numerical parameter, avoids zero arguments in log
        :param output_filename: name of the file where to write the clusters
        :param row_clusters_init: initial row clusters
        :param col_clusters_init: initial column clusters
        :param bnd_clusters_init: initial band clusters
        """
        # Input parameters -----------------
        self.nclusters_row = nclusters_row
        self.nclusters_col = nclusters_col
        self.nclusters_bnd = nclusters_bnd
        self.conv_threshold = conv_threshold
        self.max_iterations = max_iterations
        self.nruns = nruns
        self.epsilon = epsilon
        self.output_filename = output_filename
        self.row_clusters_init = row_clusters_init
        self.col_clusters_init = col_clusters_init
        self.bnd_clusters_init = bnd_clusters_init
        # Input parameters end -------------

        # store input parameters in results object
        self.results = TriclusteringResults(**self.__dict__)

        assert Z.ndim == 3, 'Incorrect dimensionality for Z matrix'
        self.Z = Z

        if all([cls is not None for cls in [row_clusters_init,
                                            col_clusters_init,
                                            bnd_clusters_init]]):
            assert nruns == 1, 'Only nruns = 1 for given initial clusters'
            assert Z.shape == (len(bnd_clusters_init),
                               len(row_clusters_init),
                               len(col_clusters_init))

    def run_with_threads(self, nthreads=1):
        """
        Run the tri-clustering using an algorithm based on numpy + threading
        (only suitable for local runs)

        :param nthreads: number of threads
        :return: tri-clustering results
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
                                self.epsilon,
                                row_clusters_init=self.row_clusters_init,
                                col_clusters_init=self.col_clusters_init,
                                bnd_clusters_init=self.bnd_clusters_init
                                ):
                r for r in range(self.nruns)
            }
            for future in concurrent.futures.as_completed(futures):
                logger.info(f'Waiting for run {self.results.nruns_completed}')
                converged, niters, row, col, bnd, e = future.result()
                logger.info(f'Error = {e}')
                if converged:
                    logger.info(f'Run converged in {niters} iterations')
                else:
                    logger.warning(f'Run not converged in {niters} iterations')
                if self.results.error is None or e < self.results.error:
                    self.results.row_clusters = row
                    self.results.col_clusters = col
                    self.results.bnd_clusters = bnd
                    self.results.error = e
                self.results.nruns_completed += 1
        self.results.write(filename=self.output_filename)
        return self.results

    def run_with_dask(self, client=None):
        raise NotImplementedError

    def run_serial(self):
        raise NotImplementedError
