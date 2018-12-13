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

import json
import numpy as np
import pandas as pd

inspire_core_list_path = 'inspire_core_list.txt'
inspire_noncore_list_path = 'inspire_noncore_list.txt'
inspire_harvested_data_path = 'inspire_harvested_data.jsonl'
inspire_core_data_path = 'inspire_core_records.jsonl'
inspire_noncore_data_path = 'inspire_noncore_records.jsonl'
save_path = 'inspire_data.df'

with open(inspire_core_list_path, 'r') as fd:
    inspire_core_arxiv_ids = set(arxiv_id.strip() for arxiv_id in fd.readlines())
with open(inspire_noncore_list_path, 'r') as fd:
    inspire_noncore_arxiv_ids = set(arxiv_id.strip() for arxiv_id in fd.readlines())

def rejected_data(harvested_data_path):
    with open(harvested_data_path, 'r') as fd:
        for line in fd:
            try:
                record = json.loads(line)
                if not (record['arxiv_identifier'] in inspire_core_arxiv_ids) and \
                        not (record['arxiv_identifier'] in inspire_noncore_arxiv_ids):
                    del record['arxiv_identifier']
                    yield record
            except:
                continue

def core_data():
    with open(inspire_core_data_path, 'r') as fd:
        for line in fd:
            yield json.loads(line)

def noncore_data():
    with open(inspire_noncore_data_path, 'r') as fd:
        for line in fd:
            yield json.loads(line)

rejected_df = pd.DataFrame(rejected_data(inspire_harvested_data_path))
rejected_df['labels'] = 0
noncore_df = pd.DataFrame(core_data())
noncore_df['labels'] = 1
core_df = pd.DataFrame(noncore_data())
core_df['labels'] = 2

inspire_data = pd.concat([rejected_df, noncore_df, core_df], ignore_index=True)
#XXX: The below data concatenation method assumes the current title and abstrat
#     combination approach i.e. using a simple whitespace. However, it's better
#     to insert a special token such as ' <endTitle> ' instead of the single
#     whitespace. This would require training our models again, as well as making
#     changes in the predict_coreness API to make the text consistent with this.
inspire_data['text'] = inspire_data['title'] + ' ' + inspire_data['abstract']
inspire_data.to_pickle(save_path)
