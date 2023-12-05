# -*- coding: utf-8 -*-
#
# This file is part of INSPIRE.
# Copyright (C) 2016-2018 CERN.
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

FROM pytorch/pytorch:0.4_cuda9_cudnn7

COPY instance-data /opt/conda/var/inspire_classifier.app-instance/
COPY boot.sh requirements.txt /app/
WORKDIR /app
RUN apt-get update && apt-get -y install ffmpeg libglib2.0-0 libsm6 libxrender1 libxext6 build-essential git
RUN python -m pip install --upgrade pip \
    && pip install --upgrade setuptools wheel \
    && pip install --upgrade --no-deps --force-reinstall https://download.pytorch.org/whl/cpu/torch-0.3.1-cp36-cp36m-linux_x86_64.whl \
    && pip freeze \
    && pip install -r requirements.txt -v \
    && pip freeze \
    && apt-get -y remove build-essential git \
    && apt-get -y autoremove && apt-get clean
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENTRYPOINT ["./boot.sh"]
