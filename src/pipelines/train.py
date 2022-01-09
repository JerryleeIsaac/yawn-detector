import argparse
import json
import os
import time

import joblib
import pandas as pd
import yaml
from loguru import logger

from src.models.nbsvm import NBSVM


def load_data(config):
    data_file = config.get("data_file")
    logger.info(f"Loading data from {data_file}")
    compression = None
    if data_file.endswith(".gz"):
        compression = "gzip"

    data_df = pd.read_csv(data_file, compression=compression)

    return data_df


def load_model(config):
    logger.info("Loading model")
    model_type = config.get("model_type")
    if model_type != "nbsvm":
        raise Exception("Unsupported model")

    model_params = config.get("model_params")
    if not model_params:
        model_params = {}
    model = NBSVM(**model_params)

    return model


def train_model(model, data_df):
    logger.info("Training model")

    x = data_df["text"]
    y = data_df["sentiment"]

    start_ts = time.time()
    model.fit(x, y)
    end_ts = time.time()
    train_duration = end_ts - start_ts
    logger.info(f"Training took {train_duration} seconds")

    train_results = {
        "duration": train_duration,
    }

    logger.info(f"Train results: {train_results}")

    return train_results


def save_model(model, config):
    model_dir = config.get("model_dir")
    model_file = config.get("model_file")

    os.makedirs(model_dir, exist_ok=True)
    logger.info(f"Saving model to {model_file}")
    joblib.dump(model, model_file)


def save_results(train_results, config):
    os.makedirs(config.get("results_dir"), exist_ok=True)
    with open(config.get("results_file"), "w") as f:
        json.dump(train_results, f)


def main(config):

    data_df = load_data(config)
    model = load_model(config)
    train_results = train_model(model, data_df)
    save_model(model, config)
    save_results(train_results, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains model")
    parser.add_argument("--params", help="Params file", required=True)
    args = parser.parse_args()

    config = None
    with open(args.params, "r") as f:
        config = yaml.safe_load(f)["train"]

    main(config)
