import logging
import os
import warnings
from pathlib import Path

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

from inspire_classifier.core.utils import (
    export_classifier_path,
    load_encoder_path,
    save_encoder_path,
    softmax,
)

warnings.filterwarnings(
    "ignore", message="load_learner` uses Python's insecure pickle module"
)

logger = logging.getLogger(__name__)


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
        if torch.cuda.is_available() and cuda_device_id >= 0:
            torch.cuda.set_device(cuda_device_id)
        else:
            default_device(False)

        number_of_backpropagation_through_time_steps = 100

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
            bs=batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            pin_memory=True,
        )

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
        logger.info("language model training starts")
        logger.info(f"Loaded torch version: {torch.__version__}")
        self.learner.fit_one_cycle(1, 1e-2)
        self.learner.unfreeze()
        self.learner.fit_one_cycle(cycle_length, 1e-3)
        logger.info("language model training finished, saving encoder")
        save_encoder_path(self.learner, finetuned_language_model_encoder_save_path)
        logger.info("encoder saved")


class Classifier:
    def __init__(self, train=False, cuda_device_id=-1, model_path=None):
        """Classifier model for document classification.

        Args:
            train (bool): Whether the model is being initialized for training.
            Defaults to False.
            cuda_device_id (int): The CUDA device ID to use. Defaults to -1 (CPU).
            model_path (str): Path to the pre-trained model file.
        """
        if torch.cuda.is_available() and cuda_device_id >= 0:
            torch.cuda.set_device(cuda_device_id)
            self.cpu = False
        else:
            default_device(False)
            self.cpu = True

        if not train:
            self.model_path = model_path or (
                Path(__file__).parent.parent
                / "models"
                / "classifier_model"
                / "classifier.h5"
            )
            if self.model_path and os.path.exists(self.model_path):
                self.load_model()
            else:
                raise FileNotFoundError(f"Model file not found at {self.model_path}")

    def load_model(self):
        """Load the trained model from file."""
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
        self.model = load_learner(self.model_path)

    def load_training_and_validation_data(
        self, train_valid_data_dir, data_itos_path, batch_size=10
    ):
        self.dls_lm_vocab = pd.read_pickle(os.path.join(data_itos_path))

        train_valid_data = pd.read_csv(train_valid_data_dir)
        self.dataloader = TextDataLoaders.from_df(
            train_valid_data,
            label_col="label",
            text_col="text",
            valid_col="is_valid",
            is_lm=False,
            bs=batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            pin_memory=True,
            text_vocab=self.dls_lm_vocab,
        )

    def initialize_learner(
        self,
        dropout_multiplier=0.5,
        weight_decay=1e-6,
        learning_rates=None,
    ):
        if learning_rates is None:
            learning_rates = np.array([1e-4, 1e-4, 1e-4, 1e-3, 1e-2])

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
        logger.info("Core classifier model training starts")
        logger.info(f"Loaded torch version: {torch.__version__}")

        self.learner.fit_one_cycle(1, 2e-2)
        self.learner.freeze_to(-2)
        self.learner.fit_one_cycle(1, slice(1e-2 / (2.6**4), 1e-2))
        self.learner.freeze_to(-3)
        self.learner.fit_one_cycle(1, slice(5e-3 / (2.6**4), 5e-3))
        self.learner.unfreeze()
        self.learner.fit_one_cycle(cycle_length, slice(1e-3 / (2.6**4), 1e-3))
        logger.info("Core classifier model training finished")
        export_classifier_path(self.learner, trained_classifier_save_path)
        self.calculate_f1_for_validation_dataset()

    def load_trained_classifier_weights(self, trained_classifier_path):
        self.model = load_learner(trained_classifier_path, cpu=self.cpu)

    def predict(self, text, temperature=0.25):
        with self.model.no_bar(), self.model.no_logging():
            prediction_scores = self.model.predict(text)
        return softmax(prediction_scores[-1].numpy(), temp=temperature)

    def predict_coreness(self, title, abstract):
        """
        Predicts the most probable category (core, non_core, rejected)
        for a given title and abstract.

        Args:
            title (str): The title of the document.
            abstract (str): The abstract of the document.
        Returns:
            dict: A dictionary containing the predicted category and its score.

        Example:
            >>> classifier = Classifier()
            >>> title = "Search for new physics in high-energy particle collisions"
            >>> abstract = "We present results from a search for beyond..."
            >>> result = classifier.predict_coreness(title, abstract)
            >>> print(result)
            {'prediction': 'core', 'scores':
                {'rejected': 0.1, 'non_core': 0.3, 'core': 0.6}}
            }
        """
        text = title + " <ENDTITLE> " + abstract
        categories = ["rejected", "non_core", "core"]
        class_probabilities = self.predict(text)
        assert len(class_probabilities) == 3

        predicted_class = categories[np.argmax(class_probabilities)]
        output_dict = {"prediction": predicted_class}
        output_dict["scores"] = dict(zip(categories, class_probabilities, strict=False))
        return output_dict

    def calculate_f1_for_validation_dataset(self):
        predictions = self.learner.get_preds(dl=self.dataloader.valid)
        y_pred = np.argmax(np.array(predictions[0]), axis=1)
        f1_validation_score = f1_score(predictions[1], y_pred, average="micro")

        logger.info(f"Validation score (f1): {f1_validation_score}")
        return f1_validation_score
