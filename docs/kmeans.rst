K-means refinement
==================

Introduction
------------

The `Kmeans` module is an implementation of the `k-means clustering`_ to refine the results of a co-clustering or
tri-clustering calculation. This k-mean refinement allows to identify similarity patterns between co- or tri-clusters.
The following pre-defined features, computed over all elements belonging to the same co- or tri-cluster, are employed
for the k-means clustering:

#. Mean value;
#. Standard deviation;
#. Minimum value;
#. Maximum value;
#. 5th percentile;
#. 95th percentile;

The implementation, which is based on the `scikit-learn`_ package, tests a range of k values and select the optimal one
on the basis of the `Silhouette coefficient`_.

.. _scikit-learn: https://scikit-learn.org/stable/index.html
.. _Silhouette coefficient: https://en.wikipedia.org/wiki/Silhouette_(clustering)

Running the refinement
----------------------

The k-means refinement should be based on existing co- or tri-clustering results:

.. code-block:: python

    import numpy as np

    Z = np.array([[1., 1., 2., 4.],
                  [1., 1., 2., 4.],
                  [3., 3., 3., 5.]])
    row_clusters = np.array([0, 0, 1, 2])  # 3 clusters
    col_cluster = np.array([0, 0, 1])  # 2 clusters

One can then setup ``Kmeans`` in the following way:

.. code-block:: python

    from cgc.kmeans import Kmeans

    km = Kmeans(
        Z,
        clusters=(row_clusters, col_cluster),
        nclusters=(3, 2)
        k_range=range(2, 5),
        kmean_max_iter=100,
        output_filename='results.json' # JSON file where to write output
    )

Here ``k_range`` is the range of ``k`` values to investigate. If not provided, a sensible range will be setup (from 2 to
a fraction of the number of co- or tri-clusters - the optional `max_k_ratio` argument allows for additional control, see
:ref:`API<API>`). ``kmean_max_iter`` is the maximum number of iterations employed for the k-means clustering.

The ``compute`` function is then called to run the k-means refinement:

.. code-block:: python

    results = km.compute()

Results
-------

The optimal ``k`` value and the k-means-based centroids computed over the co- and tri-cluster averages are stored in the
KmeansResults object:

.. code-block:: python

    results.k_value
    results.cl_mean_centroids


.. _k-means clustering: https://en.wikipedia.org/wiki/K-means_clustering

.. _API:

API
---

.. currentmodule:: cgc.kmeans

.. autoclass:: Kmeans
    :members:
    :undoc-members:

.. autoclass:: KmeansResults
