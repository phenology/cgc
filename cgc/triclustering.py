import concurrent.futures
import json
import logging

from concurrent.futures import ThreadPoolExecutor

from . import __version__
from . import triclustering_numpy

logger = logging.getLogger(__name__)


class Triclustering(object):
    """
    Perform the tri-clustering analysis of a 3D array
    """
    def __init__(self, Z, nclusters_row, nclusters_col, nclusters_bnd,
                 conv_threshold=1.e-5, max_iterations=1, nruns=1,
                 epsilon=1.e-8, output_filename=''):
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
        """
        self.Z = Z
        self.nclusters_row = nclusters_row
        self.nclusters_col = nclusters_col
        self.nclusters_bnd = nclusters_bnd
        self.conv_threshold = conv_threshold
        self.max_iterations = max_iterations
        self.nruns = nruns
        self.epsilon = epsilon
        self.output_filename = output_filename

        self.client = None

        self.row_clusters = None
        self.col_clusters = None
        self.bnd_clusters = None
        self.error = None

        self.nruns_completed = 0

    def run_with_threads(self, nthreads=1):
        """
        Run the tri-clustering using an algorithm based on numpy + threading
        (only suitable for local runs)

        :param nthreads: number of threads
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
                                row_clusters_init=self.row_clusters,
                                col_clusters_init=self.col_clusters,
                                bnd_clusters_init=self.bnd_clusters
                                ):
                r for r in range(self.nruns)
            }
            for future in concurrent.futures.as_completed(futures):
                logger.info(f'Waiting for run {self.nruns_completed} ..')
                converged, niters, row, col, bnd, e = future.result()
                logger.info(f'Error = {e}')
                if converged:
                    logger.info(f'Run converged in {niters} iterations')
                else:
                    logger.warning(f'Run not converged in {niters} iterations')
                if self.error is None or e < self.error:
                    self.row_clusters = row
                    self.col_clusters = col
                    self.bnd_clusters = bnd
                    self.error = e
                self.nruns_completed += 1
        self._write_clusters()

    def run_with_dask(self, client=None):
        raise NotImplementedError

    def run_serial(self):
        raise NotImplementedError

    def set_initial_clusters(self, row_clusters, col_clusters, bnd_clusters):
        """
        Set initial cluster assignment

        :param row_clusters: initial row clusters
        :param col_clusters: initial column clusters
        :param bnd_clusters: initial band clusters
        """
        if (not (row_clusters is None
                 and col_clusters is None
                 and bnd_clusters is None)
                and self.nruns > 1):
            logging.warning('Multiple runs with the same cluster '
                            'initialization will be performed.')
        self.row_clusters = row_clusters
        self.col_clusters = col_clusters
        self.bnd_clusters = bnd_clusters

    def _write_clusters(self):
        if self.output_filename:
            with open(self.output_filename, 'w') as f:
                data = {
                    'cgc_version': __version__,
                    'error': self.error,
                    'row_clusters': self.row_clusters.tolist(),
                    'col_clusters': self.col_clusters.tolist(),
                    'bnd_clusters': self.bnd_clusters.tolist()
                }
                json.dump(data, f, indent=4)
