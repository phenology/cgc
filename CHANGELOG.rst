###########
Change Log
###########

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning <http://semver.org/>`_.

[Unreleased]
************

[0.2.0] - 2020-09-
******************

Added
-----

* Low-memory version for numpy-based coclustering, significantly reducing the memory footprint of the code
* Numba-accelerated version of the low-memory version of the numpy-based co-clustering

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
