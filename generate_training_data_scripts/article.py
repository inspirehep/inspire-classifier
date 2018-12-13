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

"""Workflow for processing single arXiv records harvested."""

from __future__ import absolute_import, division, print_function

from workflow.patterns.controlflow import (
    IF,
    IF_NOT,
    IF_ELSE,
)

from inspirehep.modules.workflows.tasks.refextract import extract_journal_info
from inspirehep.modules.workflows.tasks.arxiv import (
    arxiv_author_list,
    arxiv_package_download,
    arxiv_derive_inspire_categories,
    populate_arxiv_document,
)
from inspirehep.modules.workflows.tasks.actions import (
    count_reference_coreness,
    download_documents,
    is_arxiv_paper,
    is_submission,
    mark,
    normalize_journal_titles,
    populate_journal_coverage,
    populate_submission_document,
    refextract,
    save_workflow,
    validate_record,
)
from inspirehep.modules.workflows.tasks.upload import set_schema

from inspirehep.modules.workflows.tasks.makejson import makejson


ENHANCE_RECORD = [
    IF(
        is_arxiv_paper,
        [
            populate_arxiv_document,
            arxiv_package_download,
            arxiv_derive_inspire_categories,
            arxiv_author_list("authorlist2marcxml.xsl"),
        ]
    ),
    IF(
        is_submission,
        populate_submission_document,
    ),
    download_documents,
    normalize_journal_titles,
    refextract,
    count_reference_coreness,
    extract_journal_info,
    populate_journal_coverage,
]

INIT_MARKS = [
    mark('auto-approved', None),
    mark('already-in-holding-pen', None),
    mark('previously_rejected', None),
    mark('is-update', None),
    mark('stopped-matched-holdingpen-wf', None),
    mark('approved', None),
    mark('unexpected-workflow-path', None),
    save_workflow
]

PRE_PROCESSING = [
    # Make sure schema is set for proper indexing in Holding Pen
    set_schema,
    INIT_MARKS,
    validate_record('hep')
]

MAKE_JSON = [
	makejson
]


class Article(object):
    """Article ingestion workflow for Literature collection."""
    name = "HEP"
    data_type = "hep"

    workflow = (
        PRE_PROCESSING +
        ENHANCE_RECORD +
        MAKE_JSON
    )
