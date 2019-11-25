import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import math
from datetime import datetime

from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error

import utils

parser = argparse.ArgumentParser(description='kaggle ashrae energy prediction')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
args = parser.parse_args()

N_FOLDS = 5

building_meta = pd.read_csv(utils.DATA_DIR / "building_metadata.csv")
train_df = pd.read_csv(utils.DATA_DIR / "train.csv")
test_df = pd.read_csv(utils.DATA_DIR / "test.csv")
weather_train = pd.read_csv(utils.DATA_DIR / "weather_train.csv")
weather_test = pd.read_csv(utils.DATA_DIR / "weather_test.csv")


experiment_name = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')

if args.debug:
    result_dir = Path(utils.RESULTS_BASE_DIR) / ('debug-' + experiment_name)
    N_FOLDS = 2
else:
    result_dir = Path(utils.RESULTS_BASE_DIR) / experiment_name

os.mkdir(result_dir)
print(f'created: {result_dir}')

train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])
weather_test['timestamp'] = pd.to_datetime(weather_test['timestamp'])

X = pd.merge(train_df, building_meta, on='building_id', how='left')
X = pd.merge(X, weather_train, on=['site_id', 'timestamp'], how='left')
test_X = pd.merge(test_df, building_meta, on='building_id', how='left')
test_X = pd.merge(test_X, weather_test, on=['site_id', 'timestamp'], how='left')

# extract feature
X['month'] = X['timestamp'].dt.month.astype(np.int8)
X['weekofyear'] = X['timestamp'].dt.weekofyear.astype(np.int8)
X['dayofyear'] = X['timestamp'].dt.dayofyear.astype(np.int16)
X['hour'] = X['timestamp'].dt.hour.astype(np.int8)
X['dayofweek'] = X['timestamp'].dt.dayofweek.astype(np.int8)
X['day_month'] = X['timestamp'].dt.day.astype(np.int8)
X['week_month'] = X['timestamp'].dt.day / 7
X['week_month'] = X['week_month'].apply(lambda x: math.ceil(x)).astype(np.int8)

test_X['month'] = test_X['timestamp'].dt.month.astype(np.int8)
test_X['weekofyear'] = test_X['timestamp'].dt.weekofyear.astype(np.int8)
test_X['dayofyear'] = test_X['timestamp'].dt.dayofyear.astype(np.int16)
test_X['hour'] = test_X['timestamp'].dt.hour.astype(np.int8)
test_X['dayofweek'] = test_X['timestamp'].dt.dayofweek.astype(np.int8)
test_X['day_month'] = test_X['timestamp'].dt.day.astype(np.int8)
test_X['week_month'] = test_X['timestamp'].dt.day / 7
test_X['week_month'] = test_X['week_month'].apply(lambda x: math.ceil(x)).astype(np.int8)

X = X.drop(['building_id', 'timestamp'], axis=1)
test_X = test_X.drop(['building_id', 'timestamp'], axis=1)

X = pd.get_dummies(X)
test_X = pd.get_dummies(test_X)

X['meter_reading'] = np.log1p(X['meter_reading'])

default_param = {
            'nthread': -1,
            'n_estimators': 10000,
            'learning_rate': 0.02,
            'num_leaves': 34,
            'colsample_bytree': 0.9497036,
            'subsample': 0.8715623,
            'max_depth': 8,
            'reg_alpha': 0.041545473,
            'reg_lambda': 0.0735294,
            'min_split_gain': 0.0222415,
            'min_child_weight': 39.3259775,
            'silent': -1,
            'verbose': -1
}

folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1001)
y_preds = np.zeros(len(X))
for n_fold, (train_idx, val_idx) in enumerate(folds.split(X)):
    train_X, val_X = X.iloc[train_idx], X.iloc[val_idx]
    train_y, val_y = train_X['meter_reading'], val_X['meter_reading']
    train_X = train_X.drop('meter_reading', axis=1)
    val_X = val_X.drop('meter_reading', axis=1)

    model = LGBMRegressor(**default_param)
    model.fit(train_X, train_y, eval_set=[(train_X, train_y), (val_X, val_y)],
            eval_metric= 'rmse', verbose= 100, early_stopping_rounds= 200)
    y_preds[val_idx] = model.predict(val_X, num_iteration=model.best_iteration_)
    model_path = result_dir / f'lgbm_fold{n_fold}.pkl'
    utils.dump_pickle(model, model_path)

val_score = np.sqrt(mean_squared_error(y_preds, X['meter_reading']))
print(val_score)

test_preds = np.zeros(len(test_X))
for i in tqdm(range(N_FOLDS)):
    model = utils.load_pickle(result_dir / f'lgbm_fold{i}.pkl')
    test_preds += model.predict(test_X.drop(['row_id'], axis=1),
                               num_iteration=model.best_iteration_)
test_preds /= 5

sample_submission = pd.read_csv(utils.DATA_DIR / 'sample_submission.csv')
sample_submission['meter_reading'] = np.expm1(test_preds)
sample_submission.to_csv('submission.csv', index=False)


