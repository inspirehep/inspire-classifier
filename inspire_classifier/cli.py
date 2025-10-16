import os

import click
import click_spinner
import pandas as pd

from inspire_classifier.core.api import train, validate
from inspire_classifier.core.model import Classifier
from inspire_classifier.core.utils import get_classifier_model_path, get_data_path


@click.group()
def inspire_classifier():  # noqa: F811
    "INSPIRE Classifier commands"


@inspire_classifier.command("predict-coreness")
@click.argument("title", type=str, required=True, nargs=1)
@click.argument("abstract", type=str, required=True, nargs=1)
@click.option(
    "-b",
    "--base-path",
    type=click.Path(exists=True),
    required=False,
    nargs=1,
    default=os.path.join(os.getcwd(), "inspire_classifier"),
)
def predict(title, abstract, base_path):
    classifier = Classifier(model_path=get_classifier_model_path(base_path))
    with click_spinner.spinner():
        click.echo(classifier.predict_coreness(title, abstract))


@inspire_classifier.command("train")
@click.option(
    "-b",
    "--base-path",
    type=click.Path(exists=True),
    required=False,
    nargs=1,
    default=os.path.join(os.getcwd(), "inspire_classifier"),
)
@click.option("-d", "--cuda-device-id", type=int, required=False, nargs=1, default=0)
@click.option("-v", "--val-fraction", type=float, required=False, nargs=1, default=0.1)
@click.option(
    "-l", "--language-model-epochs", type=int, required=False, nargs=1, default=15
)
@click.option(
    "-lb", "--language-model-batch-size", type=int, required=False, nargs=1, default=64
)
@click.option(
    "-mwf", "--minimum-word-frequency", type=int, required=False, nargs=1, default=2
)
@click.option(
    "-mvs",
    "--maximum-vocabulary-size",
    type=int,
    required=False,
    nargs=1,
    default=60000,
)
@click.option(
    "-c", "--classifier-epochs", type=int, required=False, nargs=1, default=15
)
@click.option(
    "-cb", "--classifier-batch-size", type=int, required=False, nargs=1, default=128
)
def train_classifier(
    base_path,
    cuda_device_id,
    val_fraction,
    language_model_epochs,
    language_model_batch_size,
    minimum_word_frequency,
    maximum_vocabulary_size,
    classifier_epochs,
    classifier_batch_size,
):
    with click_spinner.spinner():
        train(
            base_path=base_path,
            cuda_device_id=cuda_device_id,
            val_fraction=val_fraction,
            language_model_batch_size=language_model_batch_size,
            minimum_word_frequency=minimum_word_frequency,
            maximum_vocabulary_size=maximum_vocabulary_size,
            language_model_cycle_length=language_model_epochs,
            classifier_batch_size=classifier_batch_size,
            classifier_cycle_length=classifier_epochs,
        )


@inspire_classifier.command("validate")
@click.option(
    "-p",
    "--dataframe-path",
    type=click.Path(exists=True),
    required=False,
    nargs=1,
    default=get_data_path(
        os.path.join(os.getcwd(), "inspire_classifier"), "test_data.df"
    ),
)
@click.option(
    "-b",
    "--base-path",
    type=click.Path(exists=True),
    required=False,
    nargs=1,
    default=os.path.join(os.getcwd(), "inspire_classifier"),
)
@click.option("-d", "--cuda-device-id", type=int, required=False, nargs=1, default=0)
@click.option(
    "-t", "--softmax-temperature", type=float, required=False, nargs=1, default=0.25
)
def validate_classifier(dataframe_path, base_path, cuda_device_id, softmax_temperature):
    df = pd.read_pickle(dataframe_path)
    validate(
        df,
        base_path=base_path,
        cuda_device_id=cuda_device_id,
        softmax_temperature=softmax_temperature,
    )
