User Manual
===========

Co-clustering
-------------

The ``coclustering`` module includes tools to perform the co-clustering of
a (2D-)matrix. The code implements the Bregman block-average co-clustering 
algorithm from Ref. [#]_ and it was inspired from the Matlab `code`_ 
by Srujana Merugu and Arindam Banerjee.

.. _code: http://www.ideal.ece.utexas.edu/software.html

For a :math:`(m\times n)` array ``Z``:

.. code-block:: python

    import numpy as np
    Z = np.random.randint(100, size=(10, 4))  # m, n = 10, 4
    Z = Z.astype('float64')

In order to run the coclustering, a ``Coclustering`` object is setup:

.. code-block:: python

    from geoclustering.coclustering import Coclustering
    
    cc = Coclustering(
        Z, 
        nclusters_row=3, 
        nclusters_col=2, 
        max_iterations=100, 
        conv_threshold=1.e-5, 
        nruns=10, 
        epsilon=1.e-8
    )

Here, the number of (maximum) row- and column-clusters have been set to 3 and 2, respectively 
(a lower number of clusters could be ultimately identified). The maximum number of iterations
is set to 100. ``conv_threshold`` sets the error convergence threshold: the iterative cycle
stops when the error of two consecutive iterations differs by less than this value (the default 
is 1.e-5). The algorithm can get trapped in a local minimum, thus multiple differently-initialised
runs need to be performed to look for the clustering that optimizes the objective function. 
``nruns`` set the total number of indepent runs. The I-divergence is employed as objective 
function and a small value of ``epsilon`` guarantees that no zero-valued arguments are 
encountered in :math:`log` function (default to 1.e-8).

Tri-clustering
--------------


K-means
-------
The `kmeans` module is an implementation of `k-mean clustering`_ to the existing clustering results. 
It classifies the clustering results based on the following six statistics of each cluster cell:

#. Mean
#. STD
#. 5th percentile
#. 95th percentile
#. maximum
#. and minimum values

A ``Kmeans`` object should be set based on the existing clustering results:

.. code-block:: python

    from geoclustering.kmeans import Kmeans

    km = Kmeans(Z=Z,
        row_clusters=cc.row_clusters,
        col_clusters=cc.col_clusters,
        n_row_clusters=cc.nclusters_row,
        n_col_clusters=cc.nclusters_col,
        kmean_n_clusters=3,
        kmean_max_iter=100)

Here we present an example based on the co-clustering object ``cc`` setup in "Co-clustering" section.
``cc.nclusters_row=k`` and  ``cc.nclusters_col=l`` are the number of row/column clusters.
``Z`` is the :math:`(m\times n)` input array for co-clustering. 
``kmean_n_clusters`` is the number of k-mean clusters. 
``kmean_max_iter`` is the maximum number of iterations.

The ``compute`` function can be called to compute the k-mean results:

.. code-block:: python

    km.compute()

The centroids of "mean" statistics is stored in:

.. code-block:: python

    km.cl_mean_centroids


        
.. _k-mean clustering: https://en.wikipedia.org/wiki/K-means_clustering

.. [#] Arindam Banerjee, Inderjit Dhillon, Joydeep Ghosh, Srujana Merugu, Dharmendra S. Modha, A Generalized Maximum Entropy Approach to Bregman Co-clustering and Matrix Approximation, Journal of Machine Learning Research 8, 1919 (2007)
