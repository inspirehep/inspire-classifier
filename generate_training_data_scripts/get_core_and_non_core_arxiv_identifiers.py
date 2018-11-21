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

"""
Get the arxiv ids of all the Core and Non-Core records in INSPIRE.
Please run the code in this snippet from within the inspirehep shell.
"""

from invenio_search import current_search_client as es
from elasticsearch.helpers import scan
import numpy as np

core = []
non_core = []

for hit in scan(es, query={"query": {"exists": {"field": "arxiv_eprints"}}, "_source": ["core", "arxiv_eprints"]},
                index='records-hep', doc_type='hep'):
    source = hit['_source']
    arxiv_eprint = source['arxiv_eprints'][0]['value']
    if source.get('core') == True:
        core.append(arxiv_eprint)
    else:
        non_core.append(arxiv_eprint)

with open('inspire_core_list.txt', 'w') as fd:
    fd.writelines("{}\n".format(arxiv_id) for arxiv_id in core)
with open('inspire_noncore_list.txt', 'w') as fd:
    fd.writelines("{}\n".format(arxiv_id) for arxiv_id in non_core)