# -*- coding: utf-8 -*-
#
# This file is part of INSPIRE.
# Copyright (C) 2014-2019 CERN.
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

from __future__ import absolute_import, division, print_function

from invenio_search import current_search_client as es
from elasticsearch.helpers import scan


core = []
non_core = []

for hit in scan(es, query={"query": {"exists": {"field": "control_number"}}, "_source": ["core", "control_number"]},
                index='records-hep', doc_type='hep'):
    source = hit['_source']
    control_number = source['control_number']
    if source.get('core') == True:
        core.append(control_number)
    else:
        non_core.append(control_number)

with open('inspire_core_recids.txt', 'w') as fd:
    fd.writelines("{}\n".format(recid) for recid in core)
with open('inspire_noncore_recids.txt', 'w') as fd:
    fd.writelines("{}\n".format(recid) for recid in non_core)

