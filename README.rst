..
    This file is part of INSPIRE.
    Copyright (C) 2014-2018 CERN.

    INSPIRE is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    INSPIRE is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with INSPIRE. If not, see <http://www.gnu.org/licenses/>.

    In applying this license, CERN does not waive the privileges and immunities
    granted to it by virtue of its status as an Intergovernmental Organization
    or submit itself to any jurisdiction.


====================
 inspire-classifier
====================

.. image:: https://travis-ci.org/inspirehep/inspire-classifier.svg?branch=master
    :target: https://travis-ci.org/inspirehep/inspire-classifier

.. image:: https://coveralls.io/repos/github/inspirehep/inspire-classifier/badge.svg?branch=master
    :target: https://coveralls.io/github/inspirehep/inspire-classifier?branch=master


About
=====

INSPIRE module aimed at automatically classifying the new papers that are added to INSPIRE, such as if they are core or not, or the arXiv category corresponding to each of them.

Run the development server with:

.. highlight:: bash
    $ FLASK_DEBUG=true FLASK_APP=inspire_classifier/app.py flask run

Example:

.. highlight:: bash
    $ curl -i http://127.0.0.1:5000/api/predict/coreness --data "{\"title\": \"Alice In Wonderland\", \"abstract\": \"The reader is conveyed to Wonderland, a world that has no apparent connection with reality...\"}"
    HTTP/1.0 200 OK
    Content-Type: application/json
    Content-Length: 52
    Server: Werkzeug/0.14.1 Python/3.6.4
    Date: Wed, 22 Aug 2018 13:00:16 GMT

    {
      "score1": 0.1,
      "score2": 0.2,
      "score3": 0.7
    }