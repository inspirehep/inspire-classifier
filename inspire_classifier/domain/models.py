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

"""Classifier Domain Models."""

import collections
import torch
import torch.optim as optim
from fastai.text import (
    accuracy,
    DataLoader,
    get_rnn_classifer,
    LanguageModelLoader,
    LanguageModelData,
    load_model,
    ModelData,
    RNN_Learner,
    T,
    TextDataset,
    TextModel,
    to_gpu,
    to_np,
    save_model,
    seq2seq_reg,
    SortishSampler,
    SortSampler,
    Variable
)
from functools import partial
from inspire_classifier.utils import FastLoadTokenizer
import numpy as np
import pickle


class LanguageModel(object):
    def __init__(self, training_data_ids_path, validation_data_ids_path, language_model_model_dir,
                 data_itos_path, cuda_device_id=0, batch_size=32, dropout_multiplier=0.7):
        torch.cuda.set_device(cuda_device_id)
        self.use_cuda = True if cuda_device_id >= 0 else False

        self.inspire_data_itos = pickle.load(open(data_itos_path, 'rb'))
        self.vocabulary_size = len(self.inspire_data_itos)

        number_of_backpropagation_through_time_steps = 70
        number_of_hidden_units = 1150
        number_of_layers = 3
        self.embedding_size = 400
        optimization_function = partial(optim.Adam, betas=(0.8, 0.99))

        training_token_ids = np.load(training_data_ids_path)
        training_token_ids = np.concatenate(training_token_ids)
        validation_token_ids = np.load(validation_data_ids_path)
        validation_token_ids = np.concatenate(validation_token_ids)

        training_dataloader = LanguageModelLoader(nums=training_token_ids, bs=batch_size,
                                                  bptt=number_of_backpropagation_through_time_steps)
        validation_dataloader = LanguageModelLoader(nums=validation_token_ids, bs=batch_size,
                                                    bptt=number_of_backpropagation_through_time_steps)
        model = LanguageModelData(path=language_model_model_dir, pad_idx=1, n_tok=self.vocabulary_size,
                                  trn_dl=training_dataloader, val_dl=validation_dataloader, bs=batch_size,
                                  bptt=number_of_backpropagation_through_time_steps)

        dropouts = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * dropout_multiplier

        self.learner = model.get_model(opt_fn=optimization_function, emb_sz=self.embedding_size,
                                       n_hid=number_of_hidden_units, n_layers=number_of_layers, dropouti=dropouts[0],
                                       dropout=dropouts[1], wdrop=dropouts[2], dropoute=dropouts[3],
                                       dropouth=dropouts[4])
        self.learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
        self.learner.clip = 0.3
        self.learner.metrics = [accuracy]

    def load_pretrained_language_model_weights(self, pretrained_language_model_path, wikitext103_itos_path):
        weights = torch.load(pretrained_language_model_path, map_location=lambda storage, loc: storage)

        encoder_weights = to_np(weights['0.encoder.weight'])
        row_m = encoder_weights.mean(0)

        wikitext103_itos = pickle.load(open(wikitext103_itos_path, 'rb'))
        wikitext103_stoi = collections.defaultdict(lambda: -1, {v: k for k, v in enumerate(wikitext103_itos)})

        nw = np.zeros((self.vocabulary_size, self.embedding_size), dtype=np.float32)
        for i, w in enumerate(self.inspire_data_itos):
            r = wikitext103_stoi[w]
            if r >= 0:
                nw[i] = encoder_weights[r]
            else:
                nw[i] = row_m

        weights['0.encoder.weight'] = T(nw, cuda=self.use_cuda)
        weights['0.encoder_with_dropout.embed.weight'] = T(np.copy(nw), cuda=self.use_cuda)
        weights['1.decoder.weight'] = T(np.copy(nw), cuda=self.use_cuda)

        self.learner.model.load_state_dict(weights)

    def train(self, finetuned_language_model_encoder_save_path, learning_rate=1e-3, weight_decay=1e-7, cycle_length=15):
        self.learner.freeze_to(-1)
        self.learner.fit(learning_rate / 2, n_cycle=1, wds=weight_decay, use_clr=(32, 2), cycle_len=1)
        self.learner.unfreeze()
        self.learner.fit(learning_rate, n_cycle=1, wds=weight_decay, use_clr=(20, 10), cycle_len=cycle_length)
        save_model(self.learner.model[0], finetuned_language_model_encoder_save_path)


class Classifier(object):
    def __init__(self, data_itos_path, cuda_device_id=0, dropout_multiplier=0.5, number_of_classes=3):
        torch.cuda.set_device(cuda_device_id)

        inspire_data_itos = pickle.load(open(data_itos_path, 'rb'))
        self.vocabulary_size = len(inspire_data_itos)
        self.inspire_data_stoi = collections.defaultdict(
            lambda: 0, {str(v): int(k) for k, v in enumerate(inspire_data_itos)})

        dropouts = np.array([0.4, 0.5, 0.05, 0.3, 0.4]) * dropout_multiplier

        number_of_back_propagation_through_time_steps = 70
        number_of_hidden_units = 1150
        number_of_layers = 3
        embedding_size = 400

        self.model = get_rnn_classifer(bptt=number_of_back_propagation_through_time_steps,
                                       max_seq=20 * number_of_back_propagation_through_time_steps,
                                       n_class=number_of_classes, n_tok=self.vocabulary_size, emb_sz=embedding_size,
                                       n_hid=number_of_hidden_units, n_layers=number_of_layers, pad_token=1,
                                       layers=[embedding_size * 3, 50, number_of_classes], drops=[dropouts[4], 0.1],
                                       dropouti=dropouts[0], wdrop=dropouts[1], dropoute=dropouts[2],
                                       dropouth=dropouts[3])

        self.tokenizer = FastLoadTokenizer()

    def load_training_and_validation_data(self, training_data_ids_path, training_data_labels_path,
                                          validation_data_ids_path, validation_data_labels_path, classifier_data_dir,
                                          batch_size=10):
        training_token_ids = np.load(training_data_ids_path)
        validation_token_ids = np.load(validation_data_ids_path)
        training_labels = np.load(training_data_labels_path)
        validation_labels = np.load(validation_data_labels_path)

        training_labels = training_labels.flatten()
        validation_labels = validation_labels.flatten()
        training_labels -= training_labels.min()
        validation_labels -= validation_labels.min()

        training_dataset = TextDataset(training_token_ids, training_labels)
        validation_dataset = TextDataset(validation_token_ids, validation_labels)
        training_data_sampler = SortishSampler(data_source=training_token_ids, key=lambda x: len(training_token_ids[x]),
                                               bs=batch_size // 2)
        validation_data_sampler = SortSampler(data_source=validation_token_ids,
                                              key=lambda x: len(validation_token_ids[x]))
        training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size // 2, transpose=True,
                                         num_workers=1, pad_idx=1, sampler=training_data_sampler)
        validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, transpose=True,
                                           num_workers=1, pad_idx=1, sampler=validation_data_sampler)
        self.model_data = ModelData(path=classifier_data_dir, trn_dl=training_dataloader, val_dl=validation_dataloader)

    def initialize_learner(self):
        optimization_function = partial(optim.Adam, betas=(0.8, 0.99))

        self.learner = RNN_Learner(data=self.model_data, models=TextModel(to_gpu(self.model)),
                                   opt_fn=optimization_function)
        self.learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
        self.learner.clip = 25.
        self.learner.metrics = [accuracy]

    def load_finetuned_language_model_weights(self, finetuned_language_model_encoder_path):
        load_model(self.learner.model[0], finetuned_language_model_encoder_path)

    def train(self, trained_classifier_save_path, learning_rates=np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-2]),
              weight_decay=1e-6, cycle_length=14):
        self.learner.freeze_to(-1)
        self.learner.fit(learning_rates, n_cycle=1, wds=weight_decay, cycle_len=1, use_clr=(8, 3))
        self.learner.freeze_to(-2)
        self.learner.fit(learning_rates, n_cycle=1, wds=weight_decay, cycle_len=1, use_clr=(8, 3))

        self.learner.unfreeze()
        self.learner.fit(learning_rates, n_cycle=1, wds=weight_decay, cycle_len=cycle_length, use_clr=(32, 10))
        save_model(self.learner.model, trained_classifier_save_path)

    def load_trained_classifier_weights(self, trained_classifier_path):
        self.model.load_state_dict(torch.load(trained_classifier_path, map_location=lambda storage, loc: storage))

    def predict(self, text):
        self.model.reset()
        self.model.eval()

        input_string = 'xbos xfld 1 ' + text
        texts = [input_string]
        tokens = self.tokenizer.proc_all(texts)
        encoded_tokens = [self.inspire_data_stoi[p] for p in tokens[0]]
        token_array = np.reshape(np.array(encoded_tokens), (-1, 1))
        token_array = Variable(torch.from_numpy(token_array))
        prediction_scores = self.model(token_array)
        prediction_scores_numpy = prediction_scores[0].data.cpu().numpy()

        return numpy_softmax(prediction_scores_numpy[0])[0]


def numpy_softmax(x):
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))
