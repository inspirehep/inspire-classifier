import os
from pathlib import Path

import pandas as pd


def train_classifier(
    text_path,
    train_test_split,
    number_of_classifier_epochs,
    number_of_language_model_epochs,
):
    df = pd.read_pickle(text_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(df["label"].value_counts())
    train_size = round(min(df["label"].value_counts()) * train_test_split)
    test_size = round(min(df["label"].value_counts()) * (1 - train_test_split))

    print(train_size)
    print(test_size)
    grouped_df = df.groupby("label", as_index=False).sample(
        n=train_size, random_state=42
    )
    test_df = df.drop(grouped_df.index)
    grouped_test_df = test_df.groupby("label", as_index=False).sample(
        n=test_size, random_state=42
    )
    test_df = grouped_test_df.reset_index(drop=True)
    df = grouped_df.reset_index(drop=True)

    package_data_dir = Path(__file__).parent.parent / "inspire_classifier" / "data"
    package_data_dir.mkdir(parents=True, exist_ok=True)

    # Save dataframes to the package data folder
    df.to_pickle(package_data_dir / "train_valid_data.df")
    test_df.to_pickle(package_data_dir / "test_data.df")

    print("-----------------")
    print("Inspire Data:")
    print(f"dataframe size: {df.shape}")
    print("categories: ")
    print(df["label"].value_counts())
    print("-----------------")
    print("Test Data:")
    print(f"dataframe size: {test_df.shape}")
    print("categories: ")
    print(test_df["label"].value_counts())
    print("-----------------")

    os.system(
        f"inspire-classifier train --classifier-epochs {number_of_classifier_epochs} \
            --language-model-epochs {number_of_language_model_epochs}"
    )
    print("training finished successfully!")
    os.system("poetry run inspire-classifier validate")


# Adjust necessary data
NUMBER_OF_CLASSIFIER_EPOCHS = 1
NUMBER_OF_LANGUAGE_MODEL_EPOCHS = 1
TRAIN_TEST_SPLIT = 0.8

train_classifier(
    os.path.join(os.getcwd(), "inspire_classifier_dataset.pkl"),
    TRAIN_TEST_SPLIT,
    NUMBER_OF_CLASSIFIER_EPOCHS,
    NUMBER_OF_LANGUAGE_MODEL_EPOCHS,
)
