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

"""Classifier Core Data Preprocessors."""


import collections
import html
import numpy as np
import os
import pandas as pd
import pickle
import re
import sklearn
from fastai.text import (
    Tokenizer,
    partition_by_cores
)


BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag
re1 = re.compile(r'  +')


def split_and_save_data_for_language_model_and_classifier(dataframe_path, language_model_data_dir, classifier_data_dir,
                                                            val_fraction = 0.1, classes = ['Rejected', 'NonCore', 'Core']):
    '''
    :param dataframe_path: The path to the pandas dataframe containing the records. The dataframe should have one
                        column containing the title and abstract text appended (title + abstract). The second
                        column should contain the label as an integer (0: Rejected, 1: Non-Core, 2: Core).
    :param val_fraction: the fraction of data to use as the validation set.
    :param classes: the ordered array of prediction labels corresponding to the labels.
    '''
    inspire_data = pd.read_pickle(dataframe_path)

    # Shuffle the data
    inspire_data = inspire_data.sample(frac=1).reset_index(drop=True)
    inspire_data.columns = ['text', 'labels']
    # Swap the columns so that the labels are Column 0 and the text is Column 1
    inspire_data = inspire_data[['labels', 'text']]

    df_trn, df_val = sklearn.model_selection.train_test_split(
        inspire_data, test_size = val_fraction)

    df_trn = df_trn.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    # Save the data for the classifier
    df_trn.to_csv(os.path.join(classifier_data_dir, 'train.csv'), header=False, index=False)
    df_val.to_csv(os.path.join(classifier_data_dir, 'val.csv'), header=False, index=False)

    trn_texts = np.array(df_trn['text'])
    val_texts = np.array(df_val['text'])

    col_names = ['labels', 'text']
    df_trn = pd.DataFrame({'text': trn_texts, 'labels': [0] * len(trn_texts)}, columns=col_names)
    df_val = pd.DataFrame({'text': val_texts, 'labels': [0] * len(val_texts)}, columns=col_names)

    df_trn.to_csv(os.path.join(language_model_data_dir, 'train.csv'), header=False, index=False)
    df_val.to_csv(os.path.join(language_model_data_dir, 'val.csv'), header=False, index=False)


def generate_and_save_language_model_tokens(language_model_data_dir):
    df_trn = pd.read_csv(os.path.join(language_model_data_dir, 'train.csv'), header=None)
    df_val = pd.read_csv(os.path.join(language_model_data_dir, 'val.csv'), header=None)

    tok_trn, trn_labels = get_texts(df_trn)
    tok_val, val_labels = get_texts(df_val)

    assert len(tok_trn) == len(df_trn)

    np.save(os.path.join(language_model_data_dir, 'tok_trn.npy'), tok_trn)
    np.save(os.path.join(language_model_data_dir, 'tok_val.npy'), tok_val)
    np.save(os.path.join(language_model_data_dir, 'lbl_trn.npy'), trn_labels)
    np.save(os.path.join(language_model_data_dir, 'lbl_val.npy'), val_labels)


def map_and_save_tokens_to_ids_for_language_model(language_model_data_dir, data_itos_path, max_vocab_size = 60000,
                                                  minimum_frequency = 2):
    '''
    :param language_model_data_dir:
    :param max_vocab_size: The maximum size of the vocabulary (default: 60000)
    :param minimum_frequency: The minimum frequency that a word has to occur to be included in the vocabulary. This
                            prevents including words which occur rarely (default: 2)
    '''
    trn_tok = np.load(os.path.join(language_model_data_dir, 'tok_trn.npy'))
    val_tok = np.load(os.path.join(language_model_data_dir, 'tok_val.npy'))

    freq = collections.Counter(p for o in trn_tok for p in o)
    itos = [o for o, c in freq.most_common(max_vocab_size) if c > minimum_frequency]
    itos.insert(0, '_pad_')
    itos.insert(0, '_unk_')
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})

    trn_lm = np.array([[stoi[o] for o in p] for p in trn_tok])
    val_lm = np.array([[stoi[o] for o in p] for p in val_tok])

    np.save(os.path.join(language_model_data_dir, 'trn_ids.npy'), trn_lm)
    np.save(os.path.join(language_model_data_dir, 'val_ids.npy'), val_lm)
    pickle.dump(itos, open(data_itos_path, 'wb'))


def generate_and_save_classifier_tokens(classifier_data_dir):
    df_trn = pd.read_csv(os.path.join(classifier_data_dir, 'train.csv'), header=None)
    df_val = pd.read_csv(os.path.join(classifier_data_dir, 'val.csv'), header=None)

    tok_trn, trn_labels = get_texts(df_trn)
    tok_val, val_labels = get_texts(df_val)

    assert len(tok_trn) == len(df_trn)

    np.save(os.path.join(classifier_data_dir, 'tok_trn.npy'), tok_trn)
    np.save(os.path.join(classifier_data_dir, 'tok_val.npy'), tok_val)
    np.save(os.path.join(classifier_data_dir, 'lbl_trn.npy'), trn_labels)
    np.save(os.path.join(classifier_data_dir, 'lbl_val.npy'), val_labels)


def map_and_save_tokens_to_ids_for_classifier(classifier_data_dir, data_itos_path):
    tok_trn = np.load(os.path.join(classifier_data_dir, 'tok_trn.npy'))
    tok_val = np.load(os.path.join(classifier_data_dir, 'tok_val.npy'))

    itos = pickle.load(open(data_itos_path, 'rb'))
    stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(itos)})

    trn_ids = np.array([[stoi[o] for o in p] for p in tok_trn])
    val_ids = np.array([[stoi[o] for o in p] for p in tok_val])

    np.save(os.path.join(classifier_data_dir, 'trn_ids.npy'), trn_ids)
    np.save(os.path.join(classifier_data_dir, 'val_ids.npy'), val_ids)


def get_texts(df):
    labels = df[0].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[1].astype(str)
    texts = list(texts.apply(fixup).values)

    tok = Tokenizer().proc_all_mp(partition_by_cores(texts))
    return tok, list(labels)


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))



