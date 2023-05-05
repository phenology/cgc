###########
Change Log
###########

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning <http://semver.org/>`_.

[Unreleased]
************

[0.7.0] - 2023-05-05
********************

Added
-----

* k-means refinement now supports user-specified features

Changed
-------

* minor naming changes in classes/parameters (kmeans vs kmean, and consistent capitalization)

Fixed
-----

* result objects are now copied on return, so running more iterations does not modify previous results

[0.6.2] - 2022-03-09
********************

Fixed
-----
* Fix `mem_estimate_coclustering_numpy` on Windows: default int to 32 bit could easily overflow (`#82 <https://github.com/phenology/cgc/pull/82>`_).

Changed
-------
* Test instructions have been updated, dropping the deprecated use of setuptools' test (`#80 <https://github.com/phenology/cgc/pull/80>`_)
* Docs improvements (`#78 <https://github.com/phenology/cgc/pull/78>`_ and `#79 <https://github.com/phenology/cgc/pull/79>`_)

[0.6.1] - 2021-12-17
********************

Fixed
-----
* Fixing README - to be used as long_description on PyPI

[0.6.0] - 2021-12-17
********************

Added
-----
* k-means refinement also return refined-cluster labels

Fixed
-----
* Fixed bug in calculate_cluster_features, affecting kmeans and the calculation of the tri-cluster averages for particular ordering of the dimensions
* Number of converged runs in tri-cluster is updated

Changed
-------
* Numerical parameter epsilon is removed, which should lead to some improvement in the algorithm when empty clusters are present
* The refined cluster averages are not computed anymore over co-/tri-cluster averages but over all corresponding elements
* Dropped non-Numba powered low-mem version of co-clustering

[0.5.0] - 2021-09-23
********************

Added
-----
* k-means implementation for tri-clustering
* utility functions to calculate cluster-based averages for tri-clustering

Changed
-------
* Best k value in k-means is now selected automatically using the Silhouette score

[0.4.0] - 2021-07-29
********************

Added
-----
* utility function to estimate memory peak for numpy-based coclustering
* utility function to calculate cluster-based averages
* added Dask-based tri-clustering implementation


Fixed
-----
* k-means setup is more robust with respect to setting the range of k values and the threshold on the variance
* calculation of k-means statistics is faster


Changed
-------
* new version of tri-clustering algorithm implemented, old version moved to legacy folder


[0.3.0] - 2021-04-30
********************

Fixed
-----

* Reduced memory footprint of low-memory Dask-based implementation
* Fixed error handling in high-performance Dask implementation


Changed
-------

* Dropped tests on Python 3.6, added tests for Python 3.9 (following Dask)


[0.2.1] - 2020-09-18
********************

Fixed
-----

* Solve dependency issue: fail to install requirements with `pip`


[0.2.0] - 2020-09-17
********************

Added
-----

* Low-memory version for numpy-based coclustering, significantly reducing the memory footprint of the code
* Numba-accelerated version of the low-memory version of the numpy-based co-clustering
* Results objects include input_parameters dictionary and other metadata

Fixed
-----

* Solve issue in increasingly large Dask graph for increasing iterations

Changed
-------

* Main calculator classes stores results in dedicated object

[0.1.1] - 2020-08-27
********************

Added
-----

* Cluster results of co-/tri-clustring are now serialized to a file

Fixed
-----

* Improved output
* Bug fix in selecting minimum error run in co- and tri-clustering

Changed
-------

* K-means now loop over multiple k-values

[0.1.0] - 2020-08-11
********************

Added
-----

* First version of the CGC package, including minimal docs and tests
