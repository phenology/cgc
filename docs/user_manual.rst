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



.. [#] Arindam Banerjee, Inderjit Dhillon, Joydeep Ghosh, Srujana Merugu, Dharmendra S. Modha, A Generalized Maximum Entropy Approach to Bregman Co-clustering and Matrix Approximation, Journal of Machine Learning Research 8, 1919 (2007)
