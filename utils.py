from pathlib import Path
import pandas as pd
import pickle
import os


BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = BASE_DIR / 'data'
RESULTS_BASE_DIR = BASE_DIR / 'results'
SITE_GMT_OFFSET = [-5, 0, -7, -5, -8, 0, -5, -5, -5, -6, -7, -5, 0, -6, -5, -5]
PRIMARY_USE = [
    'Education',
    'Office',
    'Public services',
    'Entertainment/public assembly',
    'Healthcare',
    'Lodging/residential',
    'Food sales and service',
    'Retail',
    'Parking',
    'Other',
    'Utility',
    'Warehouse/storage',
    'Religious worship',
    'Technology/science',
    'Manufacturing/industrial',
    'Services'
]


def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_cv_index(df):
    cv_indices = []
    train_query = '~(timestamp < "2016-03-01")'
    val_query = '(timestamp < "2016-03-01")'
    cv_indices.append((df.query(train_query).index.tolist(),
                       df.query(val_query).index.tolist()))
    train_query = '~("2016-03-01" <= timestamp < "2016-05-01")'
    val_query = '("2016-03-01" <= timestamp < "2016-05-01")'
    cv_indices.append((df.query(train_query).index.tolist(),
                       df.query(val_query).index.tolist()))
    train_query = '~("2016-05-01" <= timestamp < "2016-07-01")'
    val_query = '("2016-05-01" <= timestamp < "2016-07-01")'
    cv_indices.append((df.query(train_query).index.tolist(),
                       df.query(val_query).index.tolist()))
    train_query = '~("2016-07-01" <= timestamp < "2016-09-01")'
    val_query = '("2016-07-01" <= timestamp < "2016-09-01")'
    cv_indices.append((df.query(train_query).index.tolist(),
                       df.query(val_query).index.tolist()))
    train_query = '~("2016-09-01" <= timestamp < "2016-11-01")'
    val_query = '("2016-09-01" <= timestamp < "2016-11-01")'
    cv_indices.append((df.query(train_query).index.tolist(),
                       df.query(val_query).index.tolist()))
    train_query = '~("2016-11-01" <= timestamp < "2017-01-01")'
    val_query = '("2016-11-01" <= timestamp < "2017-01-01")'
    cv_indices.append((df.query(train_query).index.tolist(),
                       df.query(val_query).index.tolist()))

    return cv_indices
