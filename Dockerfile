# -*- coding: utf-8 -*-
#
# This file is part of INSPIRE.
# Copyright (C) 2016-2024 CERN.
#
# INSPIRE is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# INSPIRE is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with INSPIRE. If not, see <http://www.gnu.org/licenses/>.
#
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.


FROM python:3.11-slim-buster

RUN pip install poetry==1.8.3

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

WORKDIR /app

COPY pyproject.toml poetry.lock ./

# Workarround to avoid install GPU dependencies
RUN poetry remove fastai
RUN poetry install --without dev --no-root && rm -rf $POETRY_CACHE_DIR

COPY inspire_classifier inspire_classifier/
# COPY classifier/ app/instance/
RUN poetry install --without dev

# Workarround to avoid install GPU dependencies
RUN poetry run pip install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html fastai==2.7.15


CMD ["poetry", "run", "gunicorn", "-b", ":5000", "--access-logfile", "-", "--error-logfile", "-", "inspire_classifier.flask_app:app", "--timeout 90"]
