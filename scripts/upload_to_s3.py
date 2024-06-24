import hashlib
import os
import time

import boto3


def upload_model_to_s3(
    model_path,
    output_bucket,
):
    if os.path.exists(model_path) is False:
        raise IOError(f"model file not found: {model_path}")

    hash = hashlib.sha1()
    hash.update(str(time.time()).encode("utf-8"))
    hash.hexdigest()
    model_name = f"trained_classifier_model_{hash.hexdigest()}.h5"

    session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

    s3 = session.resource("s3", endpoint_url="https://s3.cern.ch")
    s3.meta.client.upload_file(model_path, output_bucket, model_name)
    print(f"model {model_name} uploaded to s3")


OUTPUT_BUCKET_NAME = "inspire-qa-classifier/data/models/classifier_model/"

upload_model_to_s3(
    os.path.join(
        os.getcwd(),
        "classifier",
        "models",
        "classifier_model",
        "trained_classifier_model.h5",
    ),
    OUTPUT_BUCKET_NAME,
)
