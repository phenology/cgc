K-means
=======

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

API
---

.. currentmodule:: cgc.kmeans

.. autoclass:: Kmeans
    :members:
    :undoc-members:

.. autoclass:: KmeansResults
