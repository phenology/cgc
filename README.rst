.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - `fair-software.nl <https://fair-software.nl>`_ recommendations
     - Badges
   * - \1. Code repository
     - |GitHub Badge|
   * - \2. License
     - |License Badge|
   * - \3. Community Registry
     - |PyPI Badge|
   * - \4. Enable Citation
     - |Zenodo Badge| |JOSS Badge|
   * - \5. Checklist
     - |CII Best Practices Badge|
   * - **Other best practices**
     -
   * - Continuous integration
     - |Python Build| |Python Publish|
   * - Documentation
     - |Documentation Status|

.. |GitHub Badge| image:: https://img.shields.io/badge/github-repo-000.svg?logo=github&labelColor=gray&color=blue
   :target: https://github.com/phenology/cgc
   :alt: GitHub Badge

.. |License Badge| image:: https://img.shields.io/github/license/phenology/cgc
   :target: https://github.com/phenology/cgc
   :alt: License Badge

.. |PyPI Badge| image:: https://img.shields.io/pypi/v/clustering-geodata-cubes.svg?colorB=blue
   :target: https://pypi.python.org/project/clustering-geodata-cubes/
   :alt: PyPI Badge

.. |Zenodo Badge| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3979172.svg
   :target: https://doi.org/10.5281/zenodo.3979172
   :alt: Zenodo Badge

.. |CII Best Practices Badge| image:: https://bestpractices.coreinfrastructure.org/projects/4167/badge
   :target: https://bestpractices.coreinfrastructure.org/projects/4167
   :alt: CII Best Practices Badge

.. |Python Build| image:: https://github.com/phenology/cgc/workflows/Build/badge.svg
   :target: https://github.com/phenology/cgc/actions?query=workflow%3A%22Build%22
   :alt: Python Build

.. |Python Publish| image:: https://github.com/phenology/cgc/workflows/Publish/badge.svg
   :target: https://github.com/phenology/cgc/actions?query=workflow%3A%22Publish%22
   :alt: Python Publish

.. |Documentation Status| image:: https://readthedocs.org/projects/cgc/badge/?version=latest
   :target: https://cgc.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status

.. |JOSS Badge| image:: https://joss.theoj.org/papers/10.21105/joss.04032/status.svg
   :target: https://doi.org/10.21105/joss.04032
   :alt: JOSS Badge

CGC: Clustering Geo-Data Cubes
==============================

The Clustering Geo-Data Cubes (CGC) package focuses on the needs of geospatial data scientists who require tools to make sense of multi-dimensional data cubes. It provides the functionality to perform **co-cluster** and **tri-cluster** analyses on both local and distributed systems.

Installation
------------

To install CGC, do:

.. code-block:: console

  pip install clustering-geodata-cubes

Alternatively, you can clone this repository and install it using `pip`:

.. code-block:: console

  git clone https://github.com/phenology/cgc.git
  cd cgc
  pip install .


In order to run tests (including coverage) install the `dev` package version:

.. code-block:: console

  git clone https://github.com/phenology/cgc.git
  cd cgc
  pip install .[dev]
  pytest -v

Documentation
-------------

The project's full API documentation can be found `online <https://cgc.readthedocs.io/en/latest/>`_. Including:

- `Co-clustering <https://cgc.readthedocs.io/en/latest/coclustering.html>`_
- `Tri-clustering <https://cgc.readthedocs.io/en/latest/triclustering.html>`_
- `K-means refinement <https://cgc.readthedocs.io/en/latest/kmeans.html>`_
- `Utility Functions <https://cgc.readthedocs.io/en/latest/utils.html>`_

Examples of CGC applications on real geo-spatial data:

- `Co-clustering application <https://cgc-tutorial.readthedocs.io/en/latest/notebooks/coclustering.html>`_
- `Tri-clustering application <https://cgc-tutorial.readthedocs.io/en/latest/notebooks/triclustering.html>`_

Tutorial
--------

The tutorial of CGC can be found  `here <https://cgc-tutorial.readthedocs.io/en/latest/index.html>`_.


Contributing
------------

If you want to contribute to the development of cgc, have a look at the `contribution guidelines`_.

.. _contribution guidelines: https://github.com/phenology/cgc/tree/master/CONTRIBUTING.rst

License
-------

Copyright (c) 2020-2023,

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Credits
-------

The code has been developed as a collaborative effort between the `ITC, University of Twente`_ and
`the Netherlands eScience Center`_ within the generalization of the project
`High spatial resolution phenological modelling at continental scales`_.

.. _ITC, University of Twente: https://www.itc.nl
.. _High spatial resolution phenological modelling at continental scales: https://research-software.nl/projects/1334
.. _the Netherlands eScience Center: https://www.esciencecenter.nl

This package was created with `Cookiecutter <https://github.com/audreyr/cookiecutter>`_ and the
`NLeSC/python-template <https://github.com/NLeSC/python-template>`_.
