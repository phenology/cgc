Tri-clustering
==============

Introduction
------------

The ``triclustering`` module provides a generalization of the co-clustering algorithm to three-dimensional arrays (see
Ref. [#]_). For geospatial data, tri-clustering analyses allow to extend the search for similarity patterns in
data cubes, thus accounting for a band dimension in addition to space and time.

.. NOTE:: 
    The search for 'blocks' in arrays with shape (``bands``, ``rows``, ``columns``) is carried out by iteratively
    optimizing the assignment of clusters in ``rows``, ``columns`` and ``bands``, in this order. The procedure is
    repeated until convergence. The final cluster assignment might, however, be influenced by the chosen order in
    which the dimensions are considered. 

Setup the Analysis
------------------

The tri-clustering analysis of an array ``Z`` is setup by creating an instance of ``Triclustering``:

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
        epsilon=1.e-8  # numerical parameter
        output_filename='results.json'  # JSON file where to write output
    )

The input arguments of ``Triclustering`` are identical to the ``Coclustering`` ones (see :doc:`coclustering`) - ``nclusters_bnd`` is the only
additional argument, which sets the maximum number of clusters along the 'band' dimension.

.. NOTE::
    The first axis of ``Z`` is assumed to represent the 'band' dimension.

Co-clustering Implementations
-----------------------------

Local (Numpy-based)
*******************

As for the co-clustering algorithm (see :doc:`coclustering`), multiple runs of the tri-clustering algorithm can be efficiently computed in
parallel using threads. In order to run the tri-clustering analysis using 4 threads:

.. code-block:: python

    results = tc.run_with_threads(nthreads=4)

Distributed (Dask-based)
************************

Also for the tri-clustering, analysis on distributed systems can be carried out using Dask (see also
:doc:`coclustering`). Once connection to a Dask cluster is setup:

.. code-block:: python

    from dask.distributed import Client

    client = Client('tcp://daskscheduler:8786')  # connect to the Dask scheduler


the tri-clustering analysis is carried out as:

.. code-block:: python

    results = tc.run_with_dask(client)

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
