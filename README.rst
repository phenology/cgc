.. list-table::
   :widths: 25 25
   :header-rows: 1

   * - fair-software.nl recommendations
     - Badges
   * - \1. Code repository
     - |GitHub Badge|
   * - \2. License
     - |License Badge|
   * - \3. Community Registry
     - |PyPI Badge|
   * - \4. Enable Citation
     - |Zenodo Badge|
   * - \5. Checklist
     - |CII Best Practices Badge|
   * - **Other best practices**
     -
   * - Continuous integration
     - |Python Build|


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

################################################################################
CGC: Clustering Geo-Data Cubes
################################################################################

Clustering Geo-Data Cubes (CGC) is a Python package to perform clustering analysis for multidimensional geospatial data.
The included tools allow the user to efficiently run tasks in parallel on local and distributed systems.


Installation
------------

To install cgc, do:

.. code-block:: console

  git clone https://github.com/phenology/cgc.git
  cd cgc
  pip install .


Run tests (including coverage) with:

.. code-block:: console

  python setup.py test


Documentation
*************

The project's full documentation can be found `here <https://cgc.readthedocs.io/en/latest/>`_.

Contributing
************

If you want to contribute to the development of cgc,
have a look at the `contribution guidelines <CONTRIBUTING.rst>`_.

License
*******

Copyright (c) 2020,

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
*******

This package was created with `Cookiecutter <https://github.com/audreyr/cookiecutter>`_ and the `NLeSC/python-template <https://github.com/NLeSC/python-template>`_.
