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
Get Core and Non-Core records starting from an earliest date from INSPIRE.
Please run the code in this snippet from within the inspirehep shell.
"""

import datetime
from invenio_db import db
from invenio_records.models import RecordMetadata
import json
from sqlalchemy import (
    and_,
    cast,
    not_,
    or_,
    type_coerce
)
from sqlalchemy.dialects.postgresql import JSONB


STARTING_DATE = datetime.datetime(2016, 1, 1, 0, 0, 0)

base_query = db.session.query(RecordMetadata).with_entities(RecordMetadata.json['titles'][0]['title'], RecordMetadata.json['abstracts'][0]['value'])
filter_by_date = RecordMetadata.created >= STARTING_DATE
has_title_and_abstract = and_(type_coerce(RecordMetadata.json, JSONB).has_key('titles'), type_coerce(RecordMetadata.json, JSONB).has_key('abstracts'))
filter_deleted_records = or_(not_(type_coerce(RecordMetadata.json, JSONB).has_key('deleted')), not_(RecordMetadata.json['deleted'] == cast(True, JSONB)))
only_literature_collection = type_coerce(RecordMetadata.json, JSONB)['_collections'].contains(['Literature'])

only_core_records = type_coerce(RecordMetadata.json, JSONB)['core'] == cast(True, JSONB)
only_noncore_records = or_(type_coerce(RecordMetadata.json, JSONB)['core'] == cast(False, JSONB), not_(type_coerce(RecordMetadata.json, JSONB).has_key('core')))

core_query_results = base_query.filter(filter_by_date, only_core_records, has_title_and_abstract, filter_deleted_records, only_literature_collection)
noncore_query_results = base_query.filter(filter_by_date, only_noncore_records, has_title_and_abstract, filter_deleted_records, only_literature_collection)

with open('inspire_core_records.jsonl', 'w') as fd:
    for title, abstract in core_query_results:
        fd.write(json.dumps({
            'title': title,
            'abstract': abstract,
        }) + '\n')
with open('inspire_noncore_records.jsonl', 'w') as fd:
    for title, abstract in noncore_query_results:
        fd.write(json.dumps({
            'title': title,
            'abstract': abstract,
        }) + '\n')
