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

"""Classifier Domain Preprocessors."""

import pandas as pd
from fastai.text.all import RandomSplitter, range_of


def split_and_save_data_for_training(dataframe_path, dest_dir, val_fraction=0.1):
    """
    Args:
        dataframe_path: The path to the pandas dataframe containing the records.
                        The dataframe should have one column containing the title and
                        abstract text appended (title + abstract). The second column
                         should contain the label as an integer
                         (0: Rejected, 1: Non-Core, 2: Core).
        dest_dir: Directory to save the training/validation csv.
        val_fraction: the fraction of data to use as the validation set.
    """
    inspire_data = pd.read_pickle(dataframe_path)

    # Shuffle the data
    inspire_data = inspire_data.sample(frac=1).reset_index(drop=True)
    splits = RandomSplitter(valid_pct=val_fraction, seed=42)(range_of(inspire_data))

    # Add is_valid column based on the splits
    inspire_data["is_valid"] = False
    inspire_data.loc[splits[1], "is_valid"] = True

    # Save the data for the classifier and language model
    inspire_data.to_csv(dest_dir, header=True, index=False)
