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

import datetime
from inspire_dojson.utils import get_recid_from_ref
from inspirehep.utils.record_getter import (
    get_db_record,
    RecordGetterError
)
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
inspire_core_recids_path = 'inspire_core_recids.txt'
inspire_noncore_recids_path = 'inspire_noncore_recids.txt'


with open(inspire_core_recids_path, 'r') as fd:
    core_recids = set(int(recid.strip()) for recid in fd.readlines())
with open(inspire_noncore_recids_path, 'r') as fd:
    noncore_recids = set(int(recid.strip()) for recid in fd.readlines())


def get_first_order_core_noncore_reference_fractions(references):
    num_core_refs = 0
    num_noncore_refs = 0
    if references:
        for reference in references:
            recid = get_recid_from_ref(reference.get('record'))
            if recid in core_recids:
                num_core_refs += 1
            elif recid in noncore_recids:
                num_noncore_refs += 1
        total_first_order_references = len(references)
        core_references_fraction = num_core_refs / total_first_order_references
        noncore_references_fraction = num_noncore_refs / total_first_order_references
    else:
        core_references_fraction, noncore_references_fraction = 0.0, 0.0
        total_first_order_references = 0

    return core_references_fraction, noncore_references_fraction, total_first_order_references


def get_second_order_core_noncore_reference_fractions(references):
    num_core_refs = 0
    num_noncore_refs = 0
    total_second_order_references = 0
    first_order_recids = get_references_recids(references)
    missing_recids = set()
    if first_order_recids:
        for f_recid in first_order_recids:
            if not f_recid in missing_recids:
                try:
                    second_order_references = get_db_record('lit', f_recid).get('references')
                except RecordGetterError:
                    missing_recids.add(f_recid)
                    continue
                if second_order_references:
                    total_second_order_references += len(second_order_references)
                    second_order_recids = get_references_recids(second_order_references)
                    for s_recid in second_order_recids:
                        if s_recid in core_recids:
                            num_core_refs += 1
                        elif s_recid in noncore_recids:
                            num_noncore_refs += 1
    if total_second_order_references > 0:
        core_references_fraction = num_core_refs / total_second_order_references
        noncore_references_fraction = num_noncore_refs / total_second_order_references
    else:
        core_references_fraction, noncore_references_fraction = 0.0, 0.0

    return core_references_fraction, noncore_references_fraction, total_second_order_references


def get_references_recids(references):
    recids = None
    if references:
        recids = [get_recid_from_ref(reference.get('record')) for reference in references \
                    if reference.get('record')]
    return recids

base_query = db.session.query(RecordMetadata).with_entities(RecordMetadata.json['titles'][0]['title'], RecordMetadata.json['abstracts'][0]['value'], RecordMetadata.json['references'])
filter_by_date = RecordMetadata.created >= STARTING_DATE
has_title_and_abstract = and_(type_coerce(RecordMetadata.json, JSONB).has_key('titles'), type_coerce(RecordMetadata.json, JSONB).has_key('abstracts'))
filter_deleted_records = or_(not_(type_coerce(RecordMetadata.json, JSONB).has_key('deleted')), not_(RecordMetadata.json['deleted'] == cast(True, JSONB)))
only_literature_collection = type_coerce(RecordMetadata.json, JSONB)['_collections'].contains(['Literature'])

only_core_records = type_coerce(RecordMetadata.json, JSONB)['core'] == cast(True, JSONB)
only_noncore_records = or_(type_coerce(RecordMetadata.json, JSONB)['core'] == cast(False, JSONB), not_(type_coerce(RecordMetadata.json, JSONB).has_key('core')))

core_query_results = base_query.filter(filter_by_date, only_core_records, has_title_and_abstract, filter_deleted_records, only_literature_collection)
noncore_query_results = base_query.filter(filter_by_date, only_noncore_records, has_title_and_abstract, filter_deleted_records, only_literature_collection)

with open('inspire_core_records.jsonl', 'w') as fd:
    for title, abstract, references in core_query_results:
        core_references_fraction_first_order, noncore_references_fraction_first_order, total_first_order_references = get_first_order_core_noncore_reference_fractions(references)
        core_references_fraction_second_order, noncore_references_fraction_second_order, total_second_order_references = get_second_order_core_noncore_reference_fractions(references)
        fd.write(json.dumps({
            'title': title,
            'abstract': abstract,
            'core_references_fraction_first_order': core_references_fraction_first_order,
            'noncore_references_fraction_first_order': noncore_references_fraction_first_order,
            'core_references_fraction_second_order': core_references_fraction_second_order,
            'noncore_references_fraction_second_order': noncore_references_fraction_second_order,
            'total_first_order_references': total_first_order_references,
            'total_second_order_references': total_second_order_references,
        }) + '\n')
with open('inspire_noncore_records.jsonl', 'w') as fd:
    for title, abstract, references in noncore_query_results:
        core_references_fraction_first_order, noncore_references_fraction_first_order, total_first_order_references = get_first_order_core_noncore_reference_fractions(references)
        core_references_fraction_second_order, noncore_references_fraction_second_order, total_second_order_references = get_second_order_core_noncore_reference_fractions(references)
        fd.write(json.dumps({
            'title': title,
            'abstract': abstract,
            'core_references_fraction_first_order': core_references_fraction_first_order,
            'noncore_references_fraction_first_order': noncore_references_fraction_first_order,
            'core_references_fraction_second_order': core_references_fraction_second_order,
            'noncore_references_fraction_second_order': noncore_references_fraction_second_order,
            'total_first_order_references': total_first_order_references,
            'total_second_order_references': total_second_order_references,
        }) + '\n')
