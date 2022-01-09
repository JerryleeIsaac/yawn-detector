import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def load_data(config):
    data_file = config.get("data_file")
    logger.info(f"Loading data from {data_file}")
    compression = None
    if data_file.endswith(".gz"):
        compression = "gzip"

    data_df = pd.read_csv(data_file, compression=compression)

    return data_df


def load_model(config):
    model_file = config.get("model_file")
    logger.info(f"Loading model from {model_file}")
    model = joblib.load(model_file)

    return model


def evaluate(model, data_df):
    logger.info("Evaluating model")
    x = data_df["text"]
    y = data_df["sentiment"]

    y_prob = model.predict_proba(x)

    # get precision recall curve
    precisions, recalls, thresholds = precision_recall_curve(y, y_prob)
    f1s = 2 * ((precisions * recalls) / (precisions + recalls))
    optimal_threshold = thresholds[np.argmax(f1s)]
    logger.info(f"Optimal threshold {optimal_threshold}")
    model.set_threshold(optimal_threshold)
    y_pred = model.predict(x)

    test_results = {
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
    }
    model_params = {"threshold": optimal_threshold}
    predictions_df = pd.DataFrame(columns=["actual", "predicted"])
    predictions_df["actual"] = y
    predictions_df["predicted"] = y_pred

    pr_df = pd.DataFrame(columns=["precision", "recall"])
    pr_df["precision"] = precisions
    pr_df["recall"] = recalls

    logger.info(f"Test results {test_results}")
    return test_results, model_params, predictions_df, pr_df


def save_model_params(model_params, config):
    model_params_file = config.get("model_params_file")
    logger.info(f"Saving model params to {model_params_file}")

    with open(model_params_file, "w") as f:
        json.dump(model_params, f)


def save_results(test_results, config):
    results_dir = config.get("results_dir")
    results_file = config.get("results_file")
    logger.info(f"Saving results to {results_file}")

    os.makedirs(results_dir, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(test_results, f)


def save_predictions(predictions_df, config):
    results_dir = config.get("results_dir")
    prediction_results_file = config.get("prediction_results_file")
    logger.info(f"Saving predictions to {prediction_results_file}")

    os.makedirs(results_dir, exist_ok=True)
    predictions_df.to_csv(prediction_results_file, index=False)


def save_pr_curve(pr_df, config):
    results_dir = config.get("results_dir")
    pr_curve_file = config.get("pr_curve_file")
    logger.info(f"Saving precision recall curve to {pr_curve_file}")

    os.makedirs(results_dir, exist_ok=True)
    pr_df.to_csv(pr_curve_file, index=False)


def main(config):

    data_df = load_data(config)
    model = load_model(config)
    test_results, model_params, predictions_df, pr_df = evaluate(model, data_df)

    save_model_params(model_params, config)
    save_results(test_results, config)
    save_predictions(predictions_df, config)
    save_pr_curve(pr_df, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluates trained model")
    parser.add_argument("--params", help="Params file", required=True)
    args = parser.parse_args()

    config = None
    with open(args.params, "r") as f:
        config = yaml.safe_load(f)["evaluate"]

    main(config)
