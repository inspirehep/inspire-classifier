# -*- coding: utf-8 -*-
#
# This file is part of INSPIRE.
# Copyright (C) 2014-2024 CERN.
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
#
# Modified from the fastai library (https://github.com/fastai/fastai).

import os
import warnings

import numpy as np
import torch
from fastai.text.all import clean_raw_keys, distrib_barrier, get_model, rank_distrib


def save_encoder_path(self, path):
    """Save the encoder path to the config file."""
    encoder = get_model(self.model)[0]
    torch.save(encoder.state_dict(), path)


def load_encoder_path(model, path, device=None):
    encoder = get_model(model.model)[0]
    if device is None:
        device = model.dls.device
    if hasattr(encoder, "module"):
        encoder = encoder.module
    distrib_barrier()
    encoder.load_state_dict(clean_raw_keys(torch.load(path)))
    model.freeze()
    return model


def export_classifier_path(model, path):
    """Save the classifier path to the config file."""
    if rank_distrib():
        return  # don't export if child proc
    model._end_cleanup()
    old_dbunch = model.dls
    model.dls = model.dls.new_empty()
    state = model.opt.state_dict() if model.opt is not None else None
    model.opt = None
    with warnings.catch_warnings():
        # To avoid the warning that come from PyTorch about model not being checked
        warnings.simplefilter("ignore")
        torch.save(model, path)
    model.create_opt()
    if state is not None:
        model.opt.load_state_dict(state)
    model.dls = old_dbunch


def softmax(x, temp):
    return np.exp(np.divide(x, temp)) / np.sum(np.exp(np.divide(x, temp)))


def get_data_path(base_path, filename):
    return os.path.join(base_path, "data", filename)


def _get_model_path(base_path, model_type, filename):
    return os.path.join(base_path, "models", model_type, filename)


def get_language_model_path(base_path, filename="finetuned_language_model_encoder.h5"):
    return _get_model_path(base_path, "language_model", filename)


def get_classifier_model_path(base_path, filename="classifier.h5"):
    return _get_model_path(base_path, "classifier_model", filename)
