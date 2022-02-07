Tri-clustering
==============

Introduction
------------

The ``triclustering`` module provides a generalization of the co-clustering algorithm to three-dimensional arrays (see
Ref. [#]_). For geospatial data, tri-clustering analyses allow extending the search for similarity patterns in
data cubes, thus accounting for an extra dimension (the 'band' dimension) in addition to space and time.

Setup the Analysis
------------------

The tri-clustering analysis of a three-dimensional array ``Z``:

.. code-block:: python

    import numpy as np

    Z = np.array([[[1., 1., 2., 4.],
                   [1., 1., 2., 4.]],
                  [[5., 5., 8., 8.],
                   [5., 5., 8., 8.]],
                  [[6., 7., 8., 9.],
                   [6., 7., 9., 8.]]])

is setup by creating an instance of ``Triclustering``:

.. code-block:: python

    from cgc.triclustering import Triclustering
    
    tc = Triclustering(
        Z,  # data array (3D)
        nclusters_row=4,  # number of row clusters
        nclusters_col=3,  # number of column clusters
        nclusters_bnd=2,  # number of band clusters
        max_iterations=100,  # maximum number of iterations
        conv_threshold=1.e-5,  # error convergence threshold 
        nruns=10,  # number of differently-initialized runs
        output_filename='results.json'  # JSON file where to write output
    )

The input arguments of ``Triclustering`` are identical to the ``Coclustering`` ones (see :doc:`coclustering`) -
``nclusters_bnd`` is the only additional argument, which sets the maximum number of clusters along the 'band' dimension. 
Note that a lower number of clusters can be identified by the algorithm (some of the clusters may remain empty).

.. NOTE::
    The first axis of ``Z`` is assumed to represent the 'band' dimension.

Tri-clustering Implementations
------------------------------

Local (Numpy-based)
*******************

As for the co-clustering algorithm (see :doc:`coclustering`), multiple runs of the tri-clustering algorithm can be
efficiently computed in parallel using threads. In order to run the tri-clustering analysis using 4 threads:

.. code-block:: python

    results = tc.run_with_threads(nthreads=4)

Distributed (Dask-based)
************************

Also for the tri-clustering, analysis on distributed systems can be carried out using Dask (see also
:doc:`coclustering`). Once the connection to a Dask cluster is setup:

.. code-block:: python

    from dask.distributed import Client

    client = Client('tcp://daskscheduler:8786')  # connect to the Dask scheduler


the tri-clustering analysis is carried out as:

.. code-block:: python

    results = tc.run_with_dask(client)


Performance Comparison
**********************

`This notebook`_ presents a performance comparison of the two tri-clustering implementations for varying input data size
and number of clusters. To test the Dask implementation, we have used a local thread-based cluster with four workers.
As for :doc:`co-clustering <coclustering>`, we find the Numpy implementation to be much faster (~2 orders of magnitude)
than the Dask implementation for small datasets, where the Dask overhead dominates. However, when the system size
becomes sufficiently large and/or the number of clusters is increased, the Dask implementation leads to shorter timings.
It is important to stress here as well how the Dask implementation was not designed for improved performances, but to
handle large datasets that could not be otherwise tackled due to memory limitations.

.. _This notebook: https://github.com/phenology/cgc/blob/master/notebooks/time_profile/time_profile_triclustering.ipynb


Results
-------

The ``TriclusteringResults`` object returned by ``Triclustering.run_with_threads`` and ``Triclustering.run_with_dask``
contains the final row, column, and band cluster assignments (``results.row_clusters``, ``results.col_clusters``, and
``results.bnd_clusters``, respectively) as well as the approximation error of the tri-clustering (``results.error``).
Few other metadata are also present, including the input parameters employed to setup the analysis
(``results.input_parameters``).


API
---

.. currentmodule:: cgc.triclustering

.. autoclass:: Triclustering
    :members:
    :undoc-members:

.. autoclass:: TriclusteringResults

References
----------

.. [#] Xiaojing Wu, Raul Zurita-Milla, Emma Izquierdo Verdiguier, Menno-Jan Kraak, Triclustering Georeferenced Time
 Series for Analyzing Patterns of Intra-Annual Variability in Temperature, Annals of the American Association of
 Geographers 108, 71 (2018)
