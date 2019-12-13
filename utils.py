from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


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
        pickle.dump(obj, f, protocol=4)


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


def save_feature_importance(model, columns, path):
    df = pd.DataFrame()
    df['importance'] = np.log(model.feature_importances_)
    df.index = columns
    df.sort_values(by='importance', ascending=True, inplace=True)
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(y=df.index, width=df.importance)
    plt.savefig(path)


def compress_dataframe(df):
    result = df.copy()
    for col in result.columns:
        col_data = result[col]
        dn = col_data.dtype.name
        if dn == "object":
            result[col] = pd.to_numeric(col_data.astype(
                "category").cat.codes, downcast="integer")
        elif dn == "bool":
            result[col] = col_data.astype("int8")
        elif dn.startswith("int") or (col_data.round() == col_data).all():
            result[col] = pd.to_numeric(col_data, downcast="integer")
        else:
            result[col] = pd.to_numeric(col_data, downcast='float')
    return result


def cache_feature(feature_name):
    def _wrapper(func):
        #cache_name = os.path.join(FEATURES_DIR, feature_name+'.pkl')
        def _extract_feature(df, mode):
            if mode not in ['train', 'test']:
                raise ValueError('mode must be train or test.')
            cache_name = os.path.join(
                FEATURES_DIR, f'{mode}_{feature_name}.pkl')
            if os.path.exists(cache_name):
                print(f'{cache_name} found. loading feature cache')
                with open(cache_name, 'rb') as f:
                    feature = pickle.load(f)
                return feature
            print('extracting feature ...')
            feature = func(df)
            print(f'save feature as cache {cache_name}')
            with open(cache_name, 'wb') as f:
                pickle.dump(feature, f)
            return feature
        return _extract_feature
    return _wrapper
