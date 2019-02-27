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
#
# Modified from the fastai library (https://github.com/fastai/fastai).

from fastai.text import (
    BasicModel,
    Dataset,
    LinearBlock,
    MultiBatchRNN,
    num_cpus,
    Tokenizer
)
from flask import current_app
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import re
from spacy.lang.en import English
from spacy.symbols import ORTH
import torch
from torch import nn
import torch.nn.functional as F


def path_for(name):
    base_path = Path(current_app.config.get('CLASSIFIER_BASE_PATH') or current_app.instance_path)
    config_key = f'CLASSIFIER_{name}_PATH'.upper()

    return base_path / current_app.config[config_key]


class FastLoadTokenizer(Tokenizer):
    """
    Tokenizer which avoids redundant loading of spacy language model

    The FastAI Tokenizer class loads all the pipeline components of the spacy model which significantly increases
    loading time, especially when doing inference on CPU. This class inherits from the FastAI Tokenizer and is
    refactored to avoid redundant loading of the classifier.
    """
    def __init__(self):
        self.re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
        self.tok = English()
        for w in ('<eos>', '<bos>', '<unk>'):
            self.tok.tokenizer.add_special_case(w, [{ORTH: w}])

    def proc_all(self, ss):
        return [self.proc_text(s) for s in ss]

    def proc_all_mp(self, ss, ncpus=None):
        ncpus = ncpus or num_cpus() // 2
        with ProcessPoolExecutor(ncpus) as executor:
            return sum(executor.map(self.proc_all, ss), [])


def numpy_softmax(x):
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


class PoolingLinearClassifier(nn.Module):
    def __init__(self, layers, drops):
        super().__init__()
        self.layers = nn.ModuleList([
            LinearBlock(layers[i], layers[i + 1], drops[i]) for i in range(len(layers) - 1)
        ])
        self.ref_layers = nn.ModuleList([
            LinearBlock(6, 200, 0.0),
            LinearBlock(200, 100, 0.2)
        ])

    def pool(self, x, bs, is_max):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1,2,0), (1,)).view(bs,-1)

    def forward(self, text_input, ref_input):
        raw_outputs, outputs = text_input
        output = outputs[-1]
        sl,bs,_ = output.size()
        avgpool = self.pool(output, bs, False)
        mxpool = self.pool(output, bs, True)
        ref_x = ref_input
        for l_ref in self.ref_layers:
            ref_output = l_ref(ref_x)
            ref_x = F.relu(ref_output)
        x = torch.cat([output[-1], mxpool, avgpool, ref_output], 1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return l_x, raw_outputs, outputs


class TextPlusReferencesDataset(Dataset):
    def __init__(self, x_text, x_ref, y, backwards=False, sos=None, eos=None):
        self.x_text, self.x_ref, self.y, self.backwards, self.sos, self.eos = \
            x_text, x_ref, y, backwards, sos, eos

    def __getitem__(self, idx):
        x_text = self.x_text[idx]
        x_ref = self.x_ref[idx]
        if self.backwards: x_text = list(reversed(x_text))
        if self.eos is not None: x_text = x_text + [self.eos]
        if self.sos is not None: x_text = [self.sos] + x_text
        return np.array(x_text), x_ref, self.y[idx]

    def __len__(self):
        return len(self.x_text)


class MultiInputRNN(nn.Module):

    def __init__(self, rnn_encoder, final_classifier_layers, final_classifier_dropouts=[0.2, 0.1]):
        super(MultiInputRNN, self).__init__()
        self.text_network = rnn_encoder
        if hasattr(self.text_network, 'reset'):
            self.text_network.reset()
        self.combined_network = PoolingLinearClassifier(layers=final_classifier_layers, drops=final_classifier_dropouts)

    def forward(self, x_text, x_ref):
        text_network_output = self.text_network(x_text)
        output = self.combined_network(text_network_output, x_ref)

        return output


class TextPlusReferencesModel(BasicModel):
    def get_layer_groups(self):
        m = self.model
        return [(m.text_network.encoder, m.text_network.dropouti),
                *zip(m.text_network.rnns, m.text_network.dropouths),
                (m.combined_network)]


def get_rnn_classifier(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token, layers, drops, bidir=False,
                       dropouth=0.3, dropouti=0.5, dropoute=0.1, wdrop=0.5, qrnn=False):
    rnn_enc = MultiBatchRNN(bptt, max_seq, n_tok, emb_sz, n_hid, n_layers, pad_token=pad_token, bidir=bidir,
                            dropouth=dropouth, dropouti=dropouti, dropoute=dropoute, wdrop=wdrop, qrnn=qrnn)
    return MultiInputRNN(rnn_enc, layers, drops)