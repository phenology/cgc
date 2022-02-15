Co-clustering
=============

Introduction
------------

The ``coclustering`` module provides the functionality to perform the co-clustering analysis of a positive data matrix
with real-valued elements. The code implements the Bregman block-average co-clustering (BBAC) algorithm from Ref. [#]_
and it was inspired by the Matlab `code`_ by Srujana Merugu and Arindam Banerjee. Algorithmic differences include:
* The current implementation is independent of the order in which dimensions are provided (the column clustering
  followed the row clustering in the Matlab code).
* Empty (co-)clusters are handled without the need of an additional numerical parameters.


.. _code: http://www.ideal.ece.utexas.edu/software.html

The code was designed for geospatial applications (see the `Tutorial`_ section for some examples), where the array
dimensions typically correspond to space and time.

.. _Tutorial: https://cgc-tutorial.readthedocs.io

Setup the Analysis
------------------

For an array ``Z``:

.. code-block:: python

    import numpy as np

    Z = np.array([[1., 1., 2., 4.],
                  [1., 1., 2., 4.],
                  [3., 3., 3., 5.]])

the co-clustering analysis is setup by initializing a ``Coclustering`` object:

.. code-block:: python

    from cgc.coclustering import Coclustering
    
    cc = Coclustering(
        Z,  # data matrix
        nclusters_row=2, # number of row clusters
        nclusters_col=3,  # number of column clusters
        max_iterations=100,  # maximum number of iterations
        conv_threshold=1.e-5,  # error convergence threshold 
        nruns=10,  # number of differently-initialized runs
        output_filename='results.json'  # JSON file where to write output
    )

Here, we have set the maximum number of row and column clusters to 2 and 3, respectively. However, a lower number of
clusters can be identified by the algorithm (some of the clusters may remain empty). The algorithm entails an iterative
procedure that is considered converged when the error of two consecutive iterations differs by less than a threshold
(the default value is 1.e-5).

Multiple runs should be performed in order to limit the influence of the choice of initial cluster assignment on the
result. A numerical parameter guarantees that no zero-valued arguments are encountered in the logarithm that appears in
the I-divergence expression, which is employed as an objective function. Results are (optionally) written to a JSON file.

Co-clustering Implementations
-----------------------------

Local (Numpy-based)
*******************

The first one, based on `Numpy`_, is suitable to run the algorithm on a single machine. To make efficient use of
architectures with multi-core CPUs, the various differently-initialized co-clustering runs can be executed as multiple
threads. They are, in fact, embarrassingly parallel tasks that require no communication between each other. The
co-clustering analysis is run using e.g. 4 threads as:

.. code-block:: python

    results = cc.run_with_threads(nthreads=4)

Only one thread is spawned if the ``nthreads`` argument is not provided. This first implementation makes use of (fast)
matrix multiplications to calculate cluster-based properties, such as averages and distances. However, if ``Z``'s
dimensions are large, large auxiliary matrices need to be stored into memory, so that the memory requirement of this
implementation quickly becomes a bottleneck. This is the default Numpy-based implementation.

.. _Numpy: https://numpy.org

Local (Numpy-based), low-memory footprint
*****************************************

A second Numpy-based implementation makes use of an algorithm with a much lower memory footprint, and can be selected
with the optional flag ``low_memory``:

.. code-block:: python

    results = cc.run_with_threads(nthreads=4, low_memory=True)


Also here, only one thread is spawned if the ``nthreads`` argument is not provided. The reduced memory requirement
comes at a certain cost of performance. Fortunately, we also applied `Numba`_'s just-in-time compilation feature in the
``low_memory`` option. Thanks to this feature, the performance cost is significantly reduced.

.. _Numba: https://numba.pydata.org

Distributed (Dask-based)
************************

An alternative implementation makes use of `Dask`_ and is thus suitable to run the co-clustering algorithm on
distributed systems (e.g. on a cluster of compute nodes). Dask arrays are employed to process the data in chunks, which
are distributed across the cluster. This approach is thus suitable to tackle large matrices that do not fit the memory
of a single node.

If a Dask cluster is already running, we can connect to it and run the co-clustering analysis in the following way:

.. code-block:: python

    from dask.distributed import Client

    client = Client('tcp://daskscheduler:8786')  # connect to the Dask scheduler
    results = cc.run_with_dask(client)
    
.. _Dask: https://dask.org

Dask clusters can be run on different types of distributed systems: clusters of nodes connected by SSH, HPC systems,
Kubernetes clusters on cloud services. A local Dask cluster (``LocalCluster``) allows one to make use of the same
framework but using the local (multi-core) CPU(s). If no client is provided as argument, a default ``LocalCluster``
is instantiated and made use of (see `Dask docs`_).

.. _Dask docs: http://distributed.dask.org/en/stable/api.html#distributed.LocalCluster

In a second Dask-based implementation, the various co-clustering runs are submitted to the Dask scheduler, which
distributes them across the cluster. This implementation, which is activated by setting ``low_memory=False``, is
experimental and it typically leads to very large memory usages.

Performance Comparison
**********************

`This notebook`_ presents a performance comparison of the various co-clustering implementations for varying input
data size and number of clusters. For the system sizes considered, the default Numpy implementation is found to be x2-5
times faster that its low-memory counterpart, with exception of the smallest dataset sizes (similar timings are obtained
there). The comparison also includes timings obtained with the Dask implementation, which was tested with a local
thread-based cluster and four workers. While the overhead linked to using Dask is found to be significant for small
datasets, the gap between the Dask and the Numpy implementations narrows with increasing datasets and number of
clusters. Only for sufficiently large matrices and clusters the former is projected to become faster than the latter.
It is important to stress, however, that the Dask implementation was not designed for improved performances, but to
handle large datasets that could not be tackled due to memory limitations.

.. _This notebook: https://github.com/phenology/cgc/blob/master/notebooks/time_profile/time_profile_coclustering.ipynb

Results
-------

The ``CoclusteringResults`` object returned by ``Coclustering.run_with_threads`` and ``Coclustering.run_with_dask``
contains the final row and column cluster assignments (``results.row_clusters`` and ``results.col_clusters``,
respectively) as well as the approximation error of the co-clustering (``results.error``). Few other metadata are also
present, including the input parameters employed to setup the analysis (``results.input_parameters``).

API
---

.. currentmodule:: cgc.coclustering

.. autoclass:: Coclustering
    :members:
    :undoc-members:

.. autoclass:: CoclusteringResults

References
----------

.. [#] Arindam Banerjee, Inderjit Dhillon, Joydeep Ghosh, Srujana Merugu, Dharmendra S. Modha, A Generalized Maximum Entropy Approach to Bregman Co-clustering and Matrix Approximation, Journal of Machine Learning Research 8, 1919 (2007)
