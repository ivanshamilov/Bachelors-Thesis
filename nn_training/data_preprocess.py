import string
import socket
import pandas as pd
import numpy as np
from contextlib import contextmanager
from sklearn.base import BaseEstimator, TransformerMixin
from helpers.logger import Logger as logger


@contextmanager
def read_dataset(folder_path, csv_name, chunksize):
    try:
        logger.info(f"Reading {folder_path}/{csv_name}")
        df = pd.DataFrame()
        for chunk in pd.read_csv(f"{folder_path}/{csv_name}", low_memory=True, chunksize=chunksize):
            df = pd.concat([df, chunk], ignore_index=True)
        logger.info(f"Finished reading {folder_path}/{csv_name}")
        yield df
    finally:
        del df


class DataConverter(BaseEstimator, TransformerMixin):
    def __init__(self, str_columns, dtype, y_col):
        self.str_columns = str_columns.copy()
        self.dtype = dtype
        self.y_col = y_col

    def fit(self, X, y=None):
        self.num_columns = list(set(X.columns) ^ set(self.str_columns))
        return self

    def transform(self, X, y=None):
        index_to_drop = X.loc[X['protocol'].apply(lambda x: x[0])
            .isin(list(string.ascii_letters))].index.to_list()
        if self.y_col in self.str_columns:
            self.str_columns.remove(self.y_col)
        if index_to_drop:
            X.drop(index_to_drop, axis=0, inplace=True)
        X.drop(self.str_columns, axis=1, inplace=True)
        X[self.num_columns] = X[self.num_columns].astype(self.dtype)

        return X


def get_most_freq_dst_ports(data):
    return data['dst_port'].astype(np.int32).value_counts()[:50].index


def convert_dst_port_for_most_frequent(x, most_freq):
    result = 'unassigned'
    if x['dst_port'] in most_freq:
        try:
            result = socket.getservbyport(int(x['dst_port']), x['protocol'])
        except OSError:
            return 'unassigned'

    return result


class DataLabelizer(BaseEstimator, TransformerMixin):
    def __init__(self, protocols_map):
        self.protocols_map = protocols_map

    def fit(self, X, y=None):
        self.most_freq = get_most_freq_dst_ports(data=X)
        return self

    def transform(self, X, y=None):
        X['protocol'] = X['protocol'].astype(np.int32).map(self.protocols_map)
        X['dst_port'] = X.apply(lambda x: convert_dst_port_for_most_frequent(x, most_freq=self.most_freq), axis=1)
        return X
