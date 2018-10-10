# -*- coding: utf-8 -*-
#
# This file is part of INSPIRE.
# Copyright (C) 2014-2017 CERN.
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

"""Classifier Core ML Models."""

from __future__ import absolute_import, division, print_function

import collections
from fastai.text import (
    accuracy,
    DataLoader,
    get_rnn_classifier,
    LanguageModelLoader,
    LanguageModelData,
    load_model,
    ModelData,
    partition_by_cores,
    RNN_Learner,
    T,
    TextDataset,
    TextModel,
    to_gpu,
    to_np,
    Tokenizer,
    save_model,
    seq2seq_reg,
    SortishSampler,
    SortSampler,
    Variable
)
from functools import partial
import numpy as np
import pickle
import torch
import torch.optim as optim


class LanguageModel(object):
    def __init__(self, training_data_ids_path, validation_data_ids_path, language_model_model_dir,
                 data_itos_path, cuda_id = 0, batch_size = 32, dropmult = 0.7):

        if not hasattr(torch._C, '_cuda_setDevice'):
            cuda_id = -1
        torch.cuda.set_device(cuda_id)

        self.data_itos = pickle.load(open(data_itos_path, 'rb'))
        self.vocabulary_size = len(self.data_itos)

        number_of_backpropagation_through_time_steps = 70
        number_of_hidden_units = 1150
        number_of_layers = 3
        self.embedding_size = 400
        opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

        trn_lm = np.load(training_data_ids_path)
        trn_lm = np.concatenate(trn_lm)
        val_lm = np.load(validation_data_ids_path)
        val_lm = np.concatenate(val_lm)

        trn_dl = LanguageModelLoader(trn_lm, batch_size, number_of_backpropagation_through_time_steps)
        val_dl = LanguageModelLoader(val_lm, batch_size, number_of_backpropagation_through_time_steps)
        model = LanguageModelData(language_model_model_dir, 1, self.vocabulary_size, trn_dl, val_dl, bs = batch_size,
                               bptt = number_of_backpropagation_through_time_steps)

        drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * dropmult

        self.learner = model.get_model(opt_fn, self.embedding_size, number_of_hidden_units, number_of_layers,
            dropouti=drops[0], dropout=drops[1], wdrop=drops[2], dropoute=drops[3], dropouth=drops[4])
        self.learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
        self.learner.clip = 0.3
        self.learner.metrics = [accuracy]

    def load_pretrained_language_model_weights(self, pretrained_language_model_path, wikitext103_itos_path):
        wgts = torch.load(pretrained_language_model_path, map_location=lambda storage, loc: storage)

        encoder_weights = to_np(wgts['0.encoder.weight'])
        row_m = encoder_weights.mean(0)

        wt103_itos = pickle.load(open(wikitext103_itos_path, 'rb'))
        wt103_stoi = collections.defaultdict(lambda: -1, {v: k for k, v in enumerate(wt103_itos)})

        nw = np.zeros((self.vocabulary_size, self.embedding_size), dtype=np.float32)
        for i, w in enumerate(self.data_itos):
            r = wt103_stoi[w]
            if r >= 0:
                nw[i] = encoder_weights[r]
            else:
                nw[i] = row_m

        wgts['0.encoder.weight'] = T(nw)
        wgts['0.encoder_with_dropout.embed.weight'] = T(np.copy(nw))
        wgts['1.decoder.weight'] = T(np.copy(nw))

        self.learner.model.load_state_dict(wgts)

    def train(self, finetuned_language_model_encoder_save_path, learning_rate = 1e-3, weight_decay = 1e-7,
              cycle_length = 15):

        self.learner.unfreeze()
        self.learner.fit(learning_rate, n_cycle = 1, wds = weight_decay, use_clr=(20,10), cycle_len= cycle_length)
        save_model(self.learner.model[0], finetuned_language_model_encoder_save_path)


class Classifier(object):
    def __init__(self, training_data_ids_path, training_data_labels_path, validation_data_ids_path,
                 validation_data_labels_path, classifier_data_dir, data_itos_path, cuda_id = 0, batch_size = 9,
                 dropmult = 1.0):
        if not hasattr(torch._C, '_cuda_setDevice'):
            cuda_id = -1
        torch.cuda.set_device(cuda_id)

        itos = pickle.load(open(data_itos_path, 'rb'))
        self.vocabulary_size = len(itos)
        self.stoi = collections.defaultdict(lambda: 0, {str(v): int(k) for k, v in enumerate(itos)})

        trn_ids = np.load(training_data_ids_path)
        val_ids = np.load(validation_data_ids_path)
        trn_labels = np.load(training_data_labels_path)
        val_labels = np.load(validation_data_labels_path)

        #XXX: In the current dataset, the last example is too big to fit in the GPU memory, so we remove that
        trn_ids = trn_ids[:-1]
        trn_labels = trn_labels[:-1]

        trn_labels = trn_labels.flatten()
        val_labels = val_labels.flatten()
        trn_labels -= trn_labels.min()
        val_labels -= val_labels.min()
        number_of_classes = int(trn_labels.max()) + 1

        trn_ds = TextDataset(trn_ids, trn_labels)
        val_ds = TextDataset(val_ids, val_labels)
        trn_samp = SortishSampler(trn_ids, key=lambda x: len(trn_ids[x]), bs= batch_size // 2)
        val_samp = SortSampler(val_ids, key=lambda x: len(val_ids[x]))
        trn_dl = DataLoader(trn_ds, batch_size // 2, transpose = True, num_workers = 1, pad_idx = 1, sampler = trn_samp)
        val_dl = DataLoader(val_ds, batch_size, transpose = True, num_workers = 1, pad_idx = 1, sampler = val_samp)
        md = ModelData(classifier_data_dir, trn_dl, val_dl)

        dps = np.array([0.4, 0.5, 0.05, 0.3, 0.4]) * dropmult

        number_of_back_propagation_through_time_steps = 70
        number_of_hidden_units = 1150
        number_of_layers = 3
        embedding_size = 400
        opt_fn = partial(optim.Adam, betas=(0.8, 0.99))

        self.model = get_rnn_classifier(number_of_back_propagation_through_time_steps,
                                        20 * number_of_back_propagation_through_time_steps, number_of_classes,
                                        self.vocabulary_size, emb_sz = embedding_size, n_hid = number_of_hidden_units,
                                        n_layers = number_of_layers, pad_token = 1,
                                        layers = [embedding_size * 3, 50, number_of_classes], drops = [dps[4], 0.1],
                                        dropouti=dps[0], wdrop=dps[1], dropoute=dps[2], dropouth=dps[3])

        self.learner = RNN_Learner(md, TextModel(to_gpu(self.model)), opt_fn = opt_fn)
        self.learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
        self.learner.clip = 25.
        self.learner.metrics = [accuracy]

    def load_finetuned_language_model_weights(self, finetuned_language_model_encoder_path):
        load_model(self.learner.model[0], finetuned_language_model_encoder_path)

    def train(self, trained_classifier_save_path, learning_rates = np.array([1e-4,1e-4,1e-4,1e-3,1e-2]), weight_decay = 1e-6,
              cycle_length = 14):
        self.learner.freeze_to(-1)
        self.learner.fit(learning_rates, 1, wds = weight_decay, cycle_len = 1, use_clr = (8, 3))
        self.learner.freeze_to(-2)
        self.learner.fit(learning_rates, 1, wds = weight_decay, cycle_len = 1, use_clr = (8, 3))

        self.learner.unfreeze()

        self.learner.fit(learning_rates, 1, wds = weight_decay, cycle_len = cycle_length, use_clr = (32, 10))
        save_model(self.learner.model, trained_classifier_save_path)

    def load_trained_classifier_weights(self, trained_classifier_path):
        self.model.load_state_dict(torch.load(trained_classifier_path, map_location=lambda storage, loc: storage))

    def predict(self, text):
        self.model.reset()
        self.model.eval()

        input_str = 'xbos xfld 1 ' + text
        texts = [input_str]
        tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
        encoded = [self.stoi[p] for p in tok[0]]
        ary = np.reshape(np.array(encoded), (-1, 1))
        tensor = torch.from_numpy(ary)
        variable = Variable(tensor)
        predictions = self.model(to_gpu(variable))
        numpy_preds = predictions[0].data.cpu().numpy()

        return numpy_softmax(numpy_preds[0])[0]


def numpy_softmax(x):
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))