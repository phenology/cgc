###########
Change Log
###########

All notable changes to this project will be documented in this file.
This project adheres to `Semantic Versioning <http://semver.org/>`_.

[Unreleased]
************

[0.1.2] - 2020-08-
******************

Fixed
-----

* Solve issue in increasingly large Dask graph for increasing iterations

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
