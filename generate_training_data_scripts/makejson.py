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
from inspire_dojson.utils import get_recid_from_ref
from inspirehep.utils.record_getter import (
    get_db_record,
    RecordGetterError
)
import json


inspire_core_recids_path = 'inspire_core_recids.txt'
inspire_noncore_recids_path = 'inspire_noncore_recids.txt'

with open(inspire_core_recids_path, 'r') as fd:
    core_recids = set(int(recid.strip()) for recid in fd.readlines())
with open(inspire_noncore_recids_path, 'r') as fd:
    noncore_recids = set(int(recid.strip()) for recid in fd.readlines())


def makejson(obj, eng):
    title = obj.extra_data['source_data']['data']['titles'][0]['title']
    abstract = obj.extra_data['source_data']['data']['abstracts'][0]['value']
    arxiv_identifier = obj.extra_data['source_data']['data']['arxiv_eprints'][0]['value']
    references = obj.data['references']
    core_references_fraction_first_order, noncore_references_fraction_first_order, total_first_order_references = get_first_order_core_noncore_reference_fractions(
        references)
    core_references_fraction_second_order, noncore_references_fraction_second_order, total_second_order_references = get_second_order_core_noncore_reference_fractions(
        references)
    object_data = {"title": title,
                   "abstract": abstract,
                   "arxiv_identifier": arxiv_identifier,
                   "core_references_fraction_first_order": core_references_fraction_first_order,
                   "noncore_references_fraction_first_order": noncore_references_fraction_first_order,
                   "core_references_fraction_second_order": core_references_fraction_second_order,
                   "noncore_references_fraction_second_order": noncore_references_fraction_second_order,
                   "total_first_order_references": total_first_order_references,
                   "total_second_order_references": total_second_order_references
                   }

    with open('inspire_harvested_data.jsonl', 'a') as fd:
        json.dump(object_data, fd)
        fd.write("\n")


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