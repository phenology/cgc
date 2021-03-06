User Manual
===========

Co-clustering
-------------

The ``coclustering`` module provides the functionality to perform 
the co-clustering analysis of a 2D array. In typical geo-spatial applications
the array dimensions correspond to space and time (rows and columns, respectively). 
Looking for 'blocks' in such a matrix is thus equivalent to identify patterns 
that characterize the spatial and temporal axis, simultaneously.

The code implements the Bregman block-average co-clustering 
(BBAC) algorithm from Ref. [#]_ and it was inspired from the Matlab `code`_ 
by Srujana Merugu and Arindam Banerjee.

.. _code: http://www.ideal.ece.utexas.edu/software.html

For a :math:`(m\times n)` array ``Z``:

.. code-block:: python

    import numpy as np
    Z = np.random.randint(100, size=(10, 8))  # m, n = 10, 4
    Z = Z.astype('float64')

the co-clustering analysis is setup by initializing a 
``Coclustering`` object:

.. code-block:: python

    from cgc.coclustering import Coclustering
    
    cc = Coclustering(
        Z, 
        nclusters_row=4, 
        nclusters_col=3, 
        max_iterations=100,  # maximum number of iterations
        conv_threshold=1.e-5,  # error convergence threshold 
        nruns=10,  # number of differently-initialized runs
        epsilon=1.e-8,  # numerical parameter
        output_filename='results.json'  # JSON file where to write output
    )

Here, we have set the maximum number of row and column clusters to 4 and 3, respectively, 
but a lower number of clusters can be ultimately identified by the algorithm. 
The algorithm entails an iterative procedure that is considered converged when the error 
of two consecutive iterations differs by less than a threshold (the default value is 1.e-5). 

Multiple runs can be performed in order to limit the influence of the choice of initial 
conditions on the final cluster assignment. A numerical parameter guarantees that no 
zero-valued arguments are encountered in the logarithm that appears in the I-divergence, 
which is employed as objective function. Results are (optionally) written to a JSON file.

Different co-clustering implementations are available:

* The first one, based on `Numpy`_, is suitable to run the algorithm on a single machine. 
  To make efficient use of architectures with multi-core CPUs, the various 
  differently-initialized co-clustering runs can be executed as multiple threads. 
  They are, in fact, embarrassingly parallel tasks that require no communication 
  between each other. The co-clustering analysis is run using e.g. 4 threads as::

    results = cc.run_with_threads(nthreads=4)

  This first implementation makes use of (fast) matrix multiplications to calculate cluster-based
  properties, such as averages and distances. However, if ``Z``'s dimensions are large,
  large auxiliary matrices needs to be stored into memory, so that the memory requirement of this
  implementation quickly becomes a bottleneck.

.. _Numpy: https://numpy.org    

* A second Numpy-based implementation makes use of an algorithm with a much lower memory footprint,
  and can be selected with the optional flag ``low_memory``::

    results = cc.run_with_threads(nthreads=4, low_memory=True)

  The reduced memory requirement comes at the cost of performance. However, the performance loss of
  the low-memory algorithm can be significantly reduced by using `Numba`_'s just-in-time compilation
  feature, which can be activated with a second optional flag, ``numba_jit``::

    results = cc.run_with_threads(nthreads=4, low_memory=True, numba_jit=True)

.. _Numba: https://numba.pydata.org

* An alternative implementation makes use of `Dask`_ and is thus suitable to run the co-clustering
  algorithm on distributed systems (e.g. on a cluster of compute nodes). In this implementation, 
  the various co-clustering runs are submitted to the Dask scheduler, which distributes them 
  across the cluster. In addition, Dask arrays are employed to process the data in chunks, 
  which are also distributed across the cluster. 

  If a Dask cluster is already running, we can connect to it and run the co-clustering analysis 
  in the following way::

    from dask.distributed import Client
    client = Client('tcp://node0:8786')  # create connection to the Dask scheduler
    results = cc.run_with_dask(client)
    
.. _Dask: https://dask.org

  Dask clusters can be run on different types of distributed systems: clusters 
  of nodes connected by SSH, HPC systems, Kubernetes clusters on cloud services. 
  A local Dask cluster (``LocalCluster``) allows one to make use of the same 
  framework but using the locally available (multi-core) CPU(s). 

  The use of the Dask implementation in combination with a ``LocalCluster`` is 
  thus somewhat alternative to the Numpy + threading implementation for running 
  the co-clustering analysis on a single machine with multiple cores.  
  The latter implementation, however, has been found to be faster for this purpose, 
  presumably because of the reduced overhead.

* A second Dask-based implementation that is more memory-conservative is also available. 
  In this second approach, which might be suitable for particularly large matrices, the 
  differently-initialized co-clustering runs are executed sequentially, thus relying 
  on Dask for the only distribution of data chunks. This implementation can be selected 
  through the optional argument ``low_memory``::

    results = cc.run_with_dask(client, low_memory=True)

Ultimately, The arrays ``results.row_clusters`` and ``results.col_clusters`` (:math:`m-` and :math:`n-`
dimensional, respectively) contain the final row and column cluster assignments,
regardless of the implementation employed. ``results.error`` is the corresponding
approximation error.

Tri-clustering
--------------

The ``triclustering`` module provides the natural generalization of the 
co-clustering algorithm to 3D arrays. From the geo-spatial point
of view, tri-clustering analyses allow to extend the search for similarity
patterns in data-cubes, thus accounting for a band dimension in addition to 
space and time. 

.. NOTE:: 
    The search for 'blocks' in the 3D arrays is carried out by iteratively
    optimizing the assignment of clusters in rows (space), columns (time) and 
    bands, in this order. The procedure is repeated until convergence. The final
    cluster assignment might, however, be influenced by the chosen order in 
    which the dimensions are considered. 

The tri-clustering analysis of a :math:`(d\times m\times n)` array ``Z`` 
is setup by creating an instance of ``Triclustering``:

.. code-block:: python

    from cgc.triclustering import Triclustering
    
    cc = Triclustering(
        Z, 
        nclusters_row=4, 
        nclusters_col=3,
        nclusters_bnd=2,  
        max_iterations=100,  # maximum number of iterations
        conv_threshold=1.e-5,  # error convergence threshold 
        nruns=10,  # number of differently-initialized runs
        epsilon=1.e-8  # numerical parameter
        output_filename='results.json'  # JSON file where to write output
    )

The input arguments of ``Triclustering`` are almost identical to the
``Coclustering`` ones - ``nclusters_bnd`` is the only additional argument,
setting the maximum number of clusters along the 'band' axis. 

.. NOTE::
    The first axis of ``Z`` is assumed to represent the 'band' dimension.

As for the co-clustering algorithm, multiple runs of the tri-clustering 
algorithm can be efficiently computed in parallel using e.g. threads. 
In order to run the tri-clustering analysis using 4 threads: 

.. code-block:: python

    results = cc.run_with_threads(nthreads=4)

.. NOTE::
    A single tri-clustering implementation is currently available and based 
    on Numpy + threading.

K-means
-------
The `Kmeans` module is an implementation of `k-means clustering`_ to a co-clustering results.
In particular, `Kmeans` looks for the smallest value of ``k`` in a provided range such that the
sum of the cluster variances is smaller than a given threshold. K-means clusters are constructed
using the following six statistics calculated for each co-cluster cell:

#. Mean
#. Standard deviation
#. 5th percentile
#. 95th percentile
#. Maximum value
#. Minimum value

A ``Kmeans`` object should be set based on the existing co-clustering results:

.. code-block:: python

    from cgc.kmeans import Kmeans

    km = Kmeans(Z=Z,
        row_clusters=results.row_clusters,
        col_clusters=results.col_clusters,
        n_row_clusters=results.input_parameters['nclusters_row'],
        n_col_clusters=results.input_parameters['nclusters_col'],
        k_range=range(1, 5),
        kmean_max_iter=100,
        var_thres=2.,
        output_filename='results.json')

Here we present an example based on the results of co-clustering from the "Co-clustering" section.
``results.input_parameters['nclusters_row']`` and  ``results.input_parameters['nclusters_col']``
are the number of row/column clusters.
``Z`` is the :math:`(m\times n)` input array, also used for co-clustering.
``k_range`` is the range of ``k`` values to investigate.
``kmean_max_iter`` is the maximum number of iterations per each ``k`` value.
``var_thres`` sets the threshold for the selection of the best ``k`` value.

The ``compute`` function can be called to compute the k-means results:

.. code-block:: python

    results = km.compute()

In order to evaluate the outcome of the ``KMeans`` refinement, one can plot the
computed sum of variances as a function of ``k`` in what is usually known as elbow plot:

.. code-block:: python

    km.plot_elbow_curve()

The plot is also functional to select the value of ``var_thres``.
The optimal ``k`` value and the centroids of the "mean" statistics are stored in:

.. code-block:: python

    results.k_value
    results.cl_mean_centroids


.. _k-means clustering: https://en.wikipedia.org/wiki/K-means_clustering



.. [#] Arindam Banerjee, Inderjit Dhillon, Joydeep Ghosh, Srujana Merugu, Dharmendra S. Modha, A Generalized Maximum Entropy Approach to Bregman Co-clustering and Matrix Approximation, Journal of Machine Learning Research 8, 1919 (2007)
