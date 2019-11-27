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

import dataset
import utils
import features

parser = argparse.ArgumentParser(description='kaggle ashrae energy prediction')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
args = parser.parse_args()

N_FOLDS = 5

# building_meta = pd.read_csv(utils.DATA_DIR / "building_metadata.csv")
# train_df = pd.read_csv(utils.DATA_DIR / "train.csv")
# test_df = pd.read_csv(utils.DATA_DIR / "test.csv")
# weather_train = pd.read_csv(utils.DATA_DIR / "weather_train.csv")
# weather_test = pd.read_csv(utils.DATA_DIR / "weather_test.csv")


experiment_name = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')

if args.debug:
    result_dir = Path(utils.RESULTS_BASE_DIR) / ('debug-' + experiment_name)
    N_FOLDS = 2
else:
    result_dir = Path(utils.RESULTS_BASE_DIR) / experiment_name

os.mkdir(result_dir)
print(f'created: {result_dir}')

print('loading data ...')
train = dataset.AshraeDataset(mode='train', debug=args.debug)
test = dataset.AshraeDataset(mode='test')

# train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])
# test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

# weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])
# weather_test['timestamp'] = pd.to_datetime(weather_test['timestamp'])

# X = pd.merge(train_df, building_meta, on='building_id', how='left')
# X = pd.merge(X, weather_train, on=['site_id', 'timestamp'], how='left')
# test_X = pd.merge(test_df, building_meta, on='building_id', how='left')
# test_X = pd.merge(test_X, weather_test, on=['site_id', 'timestamp'], how='left')

# extract feature
print('extracting features ...')
x = pd.concat([train.merged_df.copy(), 
                     features.time_feature(train.merged_df)], 
                     axis=1)
test_x = pd.concat([test.merged_df.copy(),
                    features.time_feature(test.merged_df)],
                    axis=1)

x = x.drop(['building_id', 'timestamp'], axis=1)
test_x = test_x.drop(['building_id', 'timestamp'], axis=1)

x = pd.get_dummies(x)
test_x = pd.get_dummies(test_x)

x['meter_reading'] = np.log1p(x['meter_reading'])
print(x.columns)
print(x.shape)

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
y_preds = np.zeros(len(x))
for n_fold, (train_idx, val_idx) in enumerate(folds.split(x)):
    train_x, val_x = x.iloc[train_idx], x.iloc[val_idx]
    train_y, val_y = train_x['meter_reading'], val_x['meter_reading']
    train_x = train_x.drop('meter_reading', axis=1)
    val_x = val_x.drop('meter_reading', axis=1)

    model = LGBMRegressor(**default_param)
    model.fit(train_x, train_y, eval_set=[(train_x, train_y), (val_x, val_y)],
            eval_metric= 'rmse', verbose= 100, early_stopping_rounds= 200)
    y_preds[val_idx] = model.predict(val_x, num_iteration=model.best_iteration_)
    model_path = result_dir / f'lgbm_fold{n_fold}.pkl'
    utils.dump_pickle(model, model_path)

val_score = np.sqrt(mean_squared_error(y_preds, x['meter_reading']))
print(val_score)

print(test_x.columns)
print(test_x.shape)
print(set(train_x.columns) - set(test_x.columns))
print(set(test_x.columns) - set(train_x.columns))
test_preds = np.zeros(len(test_x))
for i in tqdm(range(N_FOLDS)):
    model = utils.load_pickle(result_dir / f'lgbm_fold{i}.pkl')
    test_preds += model.predict(test_x.drop(['row_id'], axis=1),
                               num_iteration=model.best_iteration_)
test_preds /= 5

sample_submission = pd.read_csv(utils.DATA_DIR / 'sample_submission.csv')
sample_submission['meter_reading'] = np.expm1(test_preds)
submit_save_path = result_dir / 'submission.csv'
sample_submission.to_csv(submit_save_path, index=False)
print(f'save to {submit_save_path}')



