K-means refinement
==================

Introduction
------------

The `kmeans` module is an implementation of the `k-means clustering`_ to refine the results of a co-clustering or
tri-clustering calculation. This k-means refinement allows identifying similarity patterns between co- or tri-clusters.
The following features, computed over all elements belonging to the same co- or tri-cluster, are employed by default
for the k-means clustering:

#. Mean value;
#. Standard deviation;
#. Minimum value;
#. Maximum value;
#. 5th percentile;
#. 95th percentile;

However, the user can customize the set of statistics computed over the clusters.
The implementation, which is based on the `scikit-learn`_ package, tests a range of k values and select the optimal one
based on the `Silhouette coefficient`_.

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

One can then setup ``KMeans`` in the following way:

.. code-block:: python

    from cgc.kmeans import KMeans

    km = KMeans(
        Z,
        clusters=(row_clusters, col_cluster),
        nclusters=(3, 2)
        k_range=range(2, 5),
        kmeans_kwargs={'init': 'random', 'n_init': 100},
        output_filename='results.json' # JSON file where to write output
    )

Here ``k_range`` is the range of ``k`` values to investigate. If not provided, a sensible range will be setup (from 2 to
a fraction of the number of co- or tri-clusters - the optional ``max_k_ratio`` argument allows for additional control,
see :ref:`API<API>`). ``kmeans_kwargs`` contains input arguments passed on to the `scikit-learn KMeans object`_ upon
initialization (here we define the initialization procedure). By using the optional argument ``statistics``, the user
can define a custom set of statistics employed for the k-means refinement (see the :ref:`API<API>`).

.. _scikit-learn KMeans object: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

The ``compute`` function is then called to run the k-means refinement:

.. code-block:: python

    results = km.compute()

Results
-------

The optimal ``k`` value and the refined cluster averages computed over all elements assigned to the co- and tri-clusters
are stored in the ``KMeansResults`` object:

.. code-block:: python

    results.k_value
    results.cluster_averages


.. _k-means clustering: https://en.wikipedia.org/wiki/K-means_clustering

.. _API:

API
---

.. currentmodule:: cgc.kmeans

.. autoclass:: KMeans
    :members:
    :undoc-members:

.. autoclass:: KMeansResults
