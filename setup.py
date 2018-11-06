# -*- coding: utf-8 -*-
#
# This file is part of INSPIRE.
# Copyright (C) 2014-2018 CERN.
#
# INSPIRE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# INSPIRE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with INSPIRE. If not, see <http://www.gnu.org/licenses/>.
#
# In applying this license, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

"""INSPIRE module aimed at automatically classifying the new papers that are added to INSPIRE, such as if they are core or not, or the arXiv category corresponding to each of them."""

from __future__ import absolute_import, division, print_function

from setuptools import find_packages, setup


url = 'https://github.com/inspirehep/inspire-classifier'

readme = open('README.rst').read()

setup_requires = [
    'autosemver~=0.0,>=0.5.2',
]

install_requires = [
    'fastai==0.7.0',
    'Flask~=1.0,>=1.0.2',
    'flask-apispec~=0.0,>=0.7.0',
    'marshmallow~=3.0.0b13,>=3.0.0b13',
    'numpy~=1.15,>=1.15.0',
    'spacy~=2.0,>=2.0.0',
    'torchtext==0.2.3'
]

docs_require = []

tests_require = [
    'flake8-future-import~=0.0,>=0.4.3',
    'mock~=2.0,>=2.0.0',
    'pytest-cov~=2.0,>=2.5.1',
    'pytest~=3.0,>=3.9.0',
]

extras_require = {
    'docs': docs_require,
    'tests': tests_require,
}

extras_require['all'] = []
for name, reqs in extras_require.items():
    if name not in ['all']:
        extras_require['all'].extend(reqs)

packages = find_packages(exclude=['docs', 'tests'])

setup(
    name='inspire-classifier',
    autosemver={
        'bugtracker_url': url + '/issues',
    },
    url=url,
    license='GPLv3',
    author='cern',
    author_email='admin@inspirehep.net',
    packages=packages,
    include_package_data=True,
    zip_safe=False,
    platforms='any',
    description=__doc__,
    long_description=readme,
    setup_requires=setup_requires,
    install_requires=install_requires,
    tests_require=tests_require,
    extras_require=extras_require,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
