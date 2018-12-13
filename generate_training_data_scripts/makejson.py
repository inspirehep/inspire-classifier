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

from __future__ import absolute_import, division, print_function
import json


def makejson(obj, eng):
    title = obj.extra_data['source_data']['data']['titles'][0]['title']
    abstract = obj.extra_data['source_data']['data']['abstracts'][0]['value']
    arxiv_identifier = obj.extra_data['source_data']['data']['arxiv_eprints'][0]['value']
    object_data = {"title": title, "abstract": abstract, "arxiv_identifier": arxiv_identifier}

    with open('./inspire_harvested_data.jsonl', 'a') as fd:
        json.dump(object_data, fd)
        fd.write("\n")
