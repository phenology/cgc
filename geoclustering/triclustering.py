import concurrent.futures
import logging

from concurrent.futures import ThreadPoolExecutor

from . import triclustering_numpy

logger = logging.getLogger(__name__)


class Triclustering(object):

    def __init__(self, Z, nclusters_row, nclusters_col, nclusters_bnd,
                 conv_threshold=1.e-5, max_iterations=1, nruns=1,
                 epsilon=1.e-8):
        """

        :param Z:
        :param nclusters_row:
        :param nclusters_col:
        :param nclusters_bnd:
        :param conv_threshold:
        :param niterations:
        :param nruns:
        :param epsilon:
        """
        self.Z = Z
        self.nclusters_row = nclusters_row
        self.nclusters_col = nclusters_col
        self.nclusters_bnd = nclusters_bnd
        self.conv_threshold = conv_threshold
        self.max_iterations = max_iterations
        self.nruns = nruns
        self.epsilon = epsilon

        self.client = None

        self.row_clusters = None
        self.col_clusters = None
        self.bnd_clusters = None
        self.error = None

    def run_with_threads(self, nthreads=1):
        """

        :param nthreads:
        :return:
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
                                self.epsilon):
                r for r in range(self.nruns)
            }
            row_min, col_min, bnd_min, e_min = None, None, None, 0.
            r = 0
            for future in concurrent.futures.as_completed(futures):
                logger.info(f'Waiting for run {r} ..')
                converged, niters, row, col, bnd, e = future.result()
                logger.info(f'Error = {e}')
                if converged:
                    logger.info(f'Run converged in {niters} iterations')
                else:
                    logger.warning(f'Run not converged in {niters} iterations')
                if e < e_min:
                    row_min, col_min, bnd_min, e_min = row, col, bnd, e
                r += 1
        self.row_clusters = row_min
        self.col_clusters = col_min
        self.bnd_clusters = bnd_min
        self.error = e_min

    def run_with_dask(self, client=None):
        raise NotImplementedError

    def run_serial(self):
        raise NotImplementedError
