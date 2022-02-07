#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os

from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit cgc/__version__.py
version = {}
with open(os.path.join(here, 'cgc', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read().split()

setup(
    name='clustering-geodata-cubes',
    version=version['__version__'],
    description="A clustering tool for geospatial applications",
    long_description=readme + '\n\n',
    author="Netherlands eScience Center",
    author_email='team-atlas@esciencecenter.nl',
    url='https://github.com/phenology/cgc',
    packages=find_packages(),
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='cgc',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    test_suite='tests',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
            'pycodestyle',
        ]
    }
)
