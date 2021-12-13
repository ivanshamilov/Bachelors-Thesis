#!/usr/bin/env python

import numpy as np
import pandas as pd
import csv
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from contextlib import contextmanager

sys.path.append("../diploma_script")
from helpers.logger import Logger as logger
from .data_preprocess import *
from .model import run_training

pd.options.mode.chained_assignment = None

protocols = {
    0: "raw",
    6: "tcp",
    17: "udp"
}


def converter(df, chunksize=100000):
    logger.info("Converting data...")
    df.columns = df.columns.str.lower().str.replace(" ", "_").str.replace("/", "_")
    num_chunks = (df.shape[0] // chunksize) + 1
    transformed_data = []
    for chunk in np.array_split(df, num_chunks):
        temp_data = DataConverter(str_columns=['timestamp', 'label'], dtype=np.float64, y_col='label').fit_transform(chunk)
        temp_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        temp_data.dropna(axis=0, inplace=True) # Dropping na values from dataframe
        transformed_data.append(temp_data)
    del chunk, temp_data
    transformed_data = pd.concat(transformed_data, axis=0, ignore_index=True)
    logger.ok("Conversation finished.")
    return transformed_data


def labelizer(df):   
    logger.info("Labelizing values...")
    transformed_data = DataLabelizer(protocols_map=protocols).fit_transform(df)
    transformed_data = pd.get_dummies(transformed_data, columns=['dst_port', 'protocol'])
    logger.ok("Labelizing finished.")
    return transformed_data


def reduce_dataset(df):
    transformed_data = []
    logger.info("Reducing the dataset...")
    df = df.query("Label not in ['SQL Injection', 'Infilteration', 'Brute Force -XSS', 'Brute Force -Web']")
    for label in df['Label'].unique():
        if label == 'Benign':
            d = df[df['Label'] == label][:45000]
        elif 'bruteforce' in label.lower():
            d = df[df['Label'] == label][:55000]
        else:
            d = df[df['Label'] == label][:35000]
        transformed_data.append(d)

    df = pd.concat(transformed_data, axis=0, ignore_index=True)
    del d, transformed_data
    logger.ok("Reduction finished.")

    return df


def prepare_data(df, chunksize=100000):
    df = reduce_dataset(df)
    df = converter(df, chunksize=chunksize)

    plt.figure(figsize=(15, 12), dpi=78)
    sns.countplot(x='label', data=df)
    plt.xticks(rotation=65, fontsize=13)
    plt.tight_layout()
    plt.show()

    transformed_data = labelizer(df)

    logger.info("Saving new headers...")
    with open("transformed_headers.csv", "w+", encoding="UTF-8") as f:
        writer = csv.writer(f)
        writer.writerow(transformed_data.columns.to_list())

    features = transformed_data.copy().sample(frac=1).reset_index(drop=True)
    labels = features.pop('label')
    del transformed_data

    return features, labels


def main_nn_handler(folder_path, filenames):
    li = []
    for filename in filenames:
        with read_dataset(folder_path=folder_path, csv_name=filename, chunksize=1e+5) as df:
            li.append(df)
            del df
    data = pd.concat(li, axis=0, ignore_index=True)
    features, labels = prepare_data(data, chunksize=100000)
    del li, data
    run_training(features, labels)
