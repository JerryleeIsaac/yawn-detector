import argparse
import os
import shutil
import tarfile

import pandas as pd
import yaml
from loguru import logger


def extract_archive(config):
    archive_file = config.get("archive_file")
    archive_dest = config.get("archive_dest")

    logger.info(f"Extracting {archive_file} to {archive_dest}")

    archive = tarfile.open(archive_file)
    archive.extractall(archive_dest)

    return archive_dest


def clean_archive_tmp(archive_dest):
    logger.info(f"Cleaning archive temp {archive_dest}")
    shutil.rmtree(archive_dest)


def get_train_test_dirs(archive_dest):
    train_dir = archive_dest + "/aclImdb" + "/train"
    test_dir = archive_dest + "/aclImdb" + "/test"

    logger.info(f"Using {train_dir}, {test_dir} as data directories")

    return train_dir, test_dir


def extract_text_data(sample_file, label):
    with open(sample_file, "r") as f:
        sample_data = f.read()

        return {"text": sample_data, "sentiment": label}


def create_df(data_dir):
    pos_dir = data_dir + "/pos"
    logger.info(f"Extracing positive samples from {pos_dir}.")

    neg_dir = data_dir + "/neg"
    logger.info(f"Extracing negative samples from {neg_dir}.")

    pos_sample_files = [
        os.path.join(pos_dir, sample_file)
        for sample_file in os.listdir(pos_dir)
        if sample_file.endswith(".txt")
    ]
    neg_sample_files = [
        os.path.join(neg_dir, sample_file)
        for sample_file in os.listdir(neg_dir)
        if sample_file.endswith(".txt")
    ]

    pos_samples = [
        extract_text_data(sample_file, 1) for sample_file in pos_sample_files
    ]
    neg_samples = [
        extract_text_data(sample_file, 0) for sample_file in neg_sample_files
    ]

    data_df = pd.DataFrame.from_dict(pos_samples + neg_samples)

    return data_df


def save_df(data_df, data_file, split_label, config):
    data_dir = config.get("data_dir")
    os.makedirs(data_dir, exist_ok=True)

    compression = None
    if data_file.endswith(".gz"):
        compression = "gzip"

    logger.info(f"Saving {split_label} to to {data_file}")
    data_df.to_csv(data_file, index=False, compression=compression)


def main(config):
    archive_dest = extract_archive(config)

    train_dir, test_dir = get_train_test_dirs(archive_dest)

    train_df = create_df(train_dir)
    test_df = create_df(test_dir)

    train_data_file = config.get("train_data")
    save_df(train_df, train_data_file, "train", config)
    test_data_file = config.get("test_data")
    save_df(test_df, test_data_file, "test", config)

    clean_archive_tmp(archive_dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Creates CSV file from raw data")
    parser.add_argument("--params", help="Params file", required=True)
    args = parser.parse_args()

    config = None
    with open(args.params, "r") as f:
        config = yaml.safe_load(f)["extract_raw"]

    main(config)
