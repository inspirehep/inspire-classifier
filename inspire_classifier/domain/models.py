import os

import numpy as np
import pandas as pd
import torch
from fastai.text.all import (
    AWD_LSTM,
    ColReader,
    ColSplitter,
    DataBlock,
    TextBlock,
    TextDataLoaders,
    accuracy,
    default_device,
    language_model_learner,
    load_learner,
    multiprocessing,
    text_classifier_learner,
)
from sklearn.metrics import f1_score

from inspire_classifier.utils import (
    export_classifier_path,
    load_encoder_path,
    save_encoder_path,
    softmax,
)


class LanguageModel(object):
    def __init__(
        self,
        train_valid_data_dir,
        data_itos_path,
        minimum_word_frequency,
        maximum_vocabulary_size,
        cuda_device_id,
        batch_size=32,
        dropout_multiplier=0.7,
        learning_rate=1e-3,
        weight_decay=1e-7,
    ):
        super(LanguageModel, self).__init__()
        if torch.cuda.is_available() and cuda_device_id:
            torch.cuda.set_device(cuda_device_id)
        else:
            default_device(False)

        number_of_backpropagation_through_time_steps = 70

        train_valid_data = pd.read_csv(train_valid_data_dir)

        dblock_lm = DataBlock(
            blocks=TextBlock.from_df(
                "text",
                is_lm=True,
                seq_len=number_of_backpropagation_through_time_steps,
                max_vocab=maximum_vocabulary_size,
                min_freq=minimum_word_frequency,
            ),
            get_x=ColReader("text"),
            splitter=ColSplitter("is_valid"),
        )

        dls_lm = dblock_lm.dataloaders(
            train_valid_data,
            bs=batch_size // 2,
            num_workers=multiprocessing.cpu_count() // 2,
            pin_memory=True,
        )

        # save vocab
        pd.to_pickle(dls_lm.vocab, data_itos_path)

        self.learner = language_model_learner(
            dls_lm,
            AWD_LSTM,
            metrics=accuracy,
            pretrained=True,
            drop_mult=dropout_multiplier,
            lr=learning_rate,
            wd=weight_decay,
        ).to_fp16()

    def train(self, finetuned_language_model_encoder_save_path, cycle_length=15):
        print("language model training starts")
        print(f"Loaded torch version: {torch.__version__}")
        self.learner.fit_one_cycle(1, 1e-2)
        self.learner.unfreeze()
        self.learner.fit_one_cycle(cycle_length, 1e-3)
        save_encoder_path(self.learner, finetuned_language_model_encoder_save_path)


class Classifier:
    def __init__(self, cuda_device_id):
        if torch.cuda.is_available() and cuda_device_id:
            torch.cuda.set_device(cuda_device_id)
            self.cpu = False
        else:
            default_device(False)
            self.cpu = True

    def load_training_and_validation_data(
        self, train_valid_data_dir, data_itos_path, batch_size=10
    ):
        self.dls_lm_vocab = pd.read_pickle(os.path.join(data_itos_path))

        train_valid_data = pd.read_csv(train_valid_data_dir)
        self.dataloader = TextDataLoaders.from_df(
            train_valid_data,
            label_col="labels",
            text_col="text",
            valid_col="is_valid",
            is_lm=False,
            bs=batch_size // 2,
            num_workers=multiprocessing.cpu_count() // 2,
            pin_memory=True,
            text_vocab=self.dls_lm_vocab,
        )

    def initialize_learner(
        self,
        dropout_multiplier=0.5,
        weight_decay=1e-6,
        learning_rates=np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-2]),
    ):
        self.learner = text_classifier_learner(
            self.dataloader,
            AWD_LSTM,
            drop_mult=dropout_multiplier,
            lr=learning_rates,
            wd=weight_decay,
            pretrained=False,
            metrics=accuracy,
        )

    def load_finetuned_language_model_weights(
        self, finetuned_language_model_encoder_path
    ):
        self.learner = load_encoder_path(
            self.learner, finetuned_language_model_encoder_path
        )

    def train(self, trained_classifier_save_path, cycle_length=14):
        print("Core classifier model training starts")
        print(f"Loaded torch version: {torch.__version__}")

        self.learner.fit_one_cycle(1, 2e-2)
        self.learner.freeze_to(-2)
        self.learner.fit_one_cycle(1, slice(1e-2 / (2.6**4), 1e-2))
        self.learner.freeze_to(-3)
        self.learner.fit_one_cycle(1, slice(5e-3 / (2.6**4), 5e-3))
        self.learner.unfreeze()
        self.learner.fit_one_cycle(cycle_length, slice(1e-3 / (2.6**4), 1e-3))

        export_classifier_path(self.learner, trained_classifier_save_path)
        self.calculate_f1_for_validation_dataset()

    def load_trained_classifier_weights(self, trained_classifier_path):
        self.model = load_learner(trained_classifier_path, cpu=self.cpu)

    def predict(self, text, temperature=0.25):
        prediction_scores = self.model.predict(text)
        return softmax(prediction_scores[-1].numpy(), temperature)

    def calculate_f1_for_validation_dataset(self):
        predictions = self.learner.get_preds(dl=self.dataloader.valid)
        y_pred = np.argmax(np.array(predictions[0]), axis=1)
        f1_validation_score = f1_score(predictions[1], y_pred, average="micro")

        print(f"Validation score (f1): {f1_validation_score}")
        return f1_validation_score
