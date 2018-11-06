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

from fastai.text import partition_by_cores
from inspire_classifier.utils import FastLoadTokenizer

TEST_TEXT = ['Pre-images of extreme points of the numerical range, and applications']


def test_fast_load_tokenizer_proc_all():
    tokenizer = FastLoadTokenizer()
    expected = [['pre', '-', 'images', 'of', 'extreme', 'points', 'of', 'the', 'numerical', 'range', ',', 'and',
                 'applications']]
    result = tokenizer.proc_all(TEST_TEXT)
    assert result == expected


def test_fast_load_tokenizer_proc_all_mp():
    tokenizer = FastLoadTokenizer()
    expected = [['pre', '-', 'images', 'of', 'extreme', 'points', 'of', 'the', 'numerical', 'range', ',', 'and',
                 'applications']]
    result = tokenizer.proc_all_mp(partition_by_cores(TEST_TEXT))
    assert result == expected
