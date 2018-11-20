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

"""Classifier Domain Preprocessors."""


import collections
from fastai.text import partition_by_cores
import html
from inspire_classifier.utils import FastLoadTokenizer
import numpy as np
import pandas as pd
import pickle
import re
import sklearn


BOS = 'xbos'  # beginning-of-sentence tag
FLD = 'xfld'  # data field tag
re1 = re.compile(r'  +')


def split_and_save_data_for_language_model_and_classifier(dataframe_path, language_model_data_dir, classifier_data_dir,
                                                          val_fraction=0.1):
    """
    Args:
        dataframe_path: The path to the pandas dataframe containing the records. The dataframe should have one
                        column containing the title and abstract text appended (title + abstract). The second
                        column should contain the label as an integer (0: Rejected, 1: Non-Core, 2: Core).
        language_model_data_dir: Directory to store language model data.
        classifier_data_dir: Directory to save classifier data.
        val_fraction: the fraction of data to use as the validation set.
    """
    inspire_data = pd.read_pickle(dataframe_path)

    # Shuffle the data
    inspire_data = inspire_data.sample(frac=1).reset_index(drop=True)
    # Swap the columns so that the labels are Column 0 and the text is Column 1 (and remove any additional columns)
    inspire_data = inspire_data[['labels', 'text']]

    training_dataframe, validation_dataframe = sklearn.model_selection.train_test_split(
        inspire_data, test_size=val_fraction)

    training_dataframe = training_dataframe.reset_index(drop=True)
    validation_dataframe = validation_dataframe.reset_index(drop=True)

    # Save the data for the classifier
    training_dataframe.to_csv(classifier_data_dir / 'training_data.csv', header=False, index=False)
    validation_dataframe.to_csv(classifier_data_dir / 'validation_data.csv', header=False, index=False)

    training_texts = np.array(training_dataframe['text'])
    validation_texts = np.array(validation_dataframe['text'])

    column_names = ['labels', 'text']
    training_dataframe_for_language_model = pd.DataFrame({'text': training_texts, 'labels': [0] * len(training_texts)},
                                                         columns=column_names)
    validation_dataframe_for_language_model = pd.DataFrame({'text': validation_texts,
                                                            'labels': [0] * len(validation_texts)},
                                                           columns=column_names)

    training_dataframe_for_language_model.to_csv(language_model_data_dir / 'training_data.csv', header=False,
                                                 index=False)
    validation_dataframe_for_language_model.to_csv(language_model_data_dir / 'validation_data.csv', header=False,
                                                   index=False)


def generate_and_save_language_model_tokens(language_model_data_dir):
    training_dataframe = pd.read_csv(language_model_data_dir / 'training_data.csv', header=None)
    validation_dataframe = pd.read_csv(language_model_data_dir / 'validation_data.csv', header=None)

    training_tokens, training_labels = get_texts(training_dataframe)
    validation_tokens, validation_labels = get_texts(validation_dataframe)

    assert len(training_tokens) == len(training_dataframe)

    np.save(language_model_data_dir / 'training_tokens.npy', training_tokens)
    np.save(language_model_data_dir / 'validation_tokens.npy', validation_tokens)
    np.save(language_model_data_dir / 'training_labels.npy', training_labels)
    np.save(language_model_data_dir / 'validation_labels.npy', validation_labels)


def map_and_save_tokens_to_ids_for_language_model(language_model_data_dir, data_itos_path, max_vocab_size=60000,
                                                  minimum_frequency=2):
    """
    Args:
        language_model_data_dir: Directory for language model data.
        data_itos_path: The path to save the data ITOS which maps the words in the vocabulary to numerical indices.
        max_vocab_size: The maximum size of the vocabulary (default: 60000).
        minimum_frequency: The minimum frequency that a word has to occur to be included in the vocabulary. This
                            prevents including words which occur rarely (default: 2)
    """
    training_tokens = np.load(language_model_data_dir / 'training_tokens.npy')
    validation_tokens = np.load(language_model_data_dir / 'validation_tokens.npy')

    word_frequency = collections.Counter(p for o in training_tokens for p in o)
    inspire_data_itos = [o for o, c in word_frequency.most_common(max_vocab_size) if c > minimum_frequency]
    inspire_data_itos.insert(0, '_pad_')
    inspire_data_itos.insert(0, '_unk_')
    inspire_data_stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(inspire_data_itos)})

    training_token_ids = np.array([[inspire_data_stoi[o] for o in p] for p in training_tokens])
    validation_token_ids = np.array([[inspire_data_stoi[o] for o in p] for p in validation_tokens])

    np.save(language_model_data_dir / 'training_token_ids.npy', training_token_ids)
    np.save(language_model_data_dir / 'validation_token_ids.npy', validation_token_ids)
    pickle.dump(inspire_data_itos, open(data_itos_path, 'wb'))


def generate_and_save_classifier_tokens(classifier_data_dir):
    training_dataframe = pd.read_csv(classifier_data_dir / 'training_data.csv', header=None)
    validation_dataframe = pd.read_csv(classifier_data_dir / 'validation_data.csv', header=None)

    training_tokens, training_labels = get_texts(training_dataframe)
    validation_tokens, validation_labels = get_texts(validation_dataframe)

    assert len(training_tokens) == len(training_dataframe)

    np.save(classifier_data_dir / 'training_tokens.npy', training_tokens)
    np.save(classifier_data_dir / 'validation_tokens.npy', validation_tokens)
    np.save(classifier_data_dir / 'training_labels.npy', training_labels)
    np.save(classifier_data_dir / 'validation_labels.npy', validation_labels)


def map_and_save_tokens_to_ids_for_classifier(classifier_data_dir, data_itos_path):
    training_tokens = np.load(classifier_data_dir / 'training_tokens.npy')
    validation_tokens = np.load(classifier_data_dir / 'validation_tokens.npy')

    inspire_data_itos = pickle.load(open(data_itos_path, 'rb'))
    inspire_data_stoi = collections.defaultdict(lambda: 0, {v: k for k, v in enumerate(inspire_data_itos)})

    training_token_ids = np.array([[inspire_data_stoi[o] for o in p] for p in training_tokens])
    validation_token_ids = np.array([[inspire_data_stoi[o] for o in p] for p in validation_tokens])

    np.save(classifier_data_dir / 'training_token_ids.npy', training_token_ids)
    np.save(classifier_data_dir / 'validation_token_ids.npy', validation_token_ids)


def get_texts(df):
    labels = df[0].values.astype(np.int64)
    texts = f'\n{BOS} {FLD} 1 ' + df[1].astype(str)
    texts = list(texts.apply(fixup).values)

    tokens = FastLoadTokenizer().proc_all_mp(partition_by_cores(texts))
    return tokens, list(labels)


def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))
