import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import math
from datetime import datetime
from logging import getLogger, StreamHandler, FileHandler, DEBUG

from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

import dataset
import utils
import features
import mylogger
import mycallbacks
import slack
import preprocessing


parser = argparse.ArgumentParser(description='kaggle ashrae energy prediction')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
args = parser.parse_args()

N_FOLDS = 6

# building_meta = pd.read_csv(utils.DATA_DIR / "building_metadata.csv")
# train_df = pd.read_csv(utils.DATA_DIR / "train.csv")
# test_df = pd.read_csv(utils.DATA_DIR / "test.csv")
# weather_train = pd.read_csv(utils.DATA_DIR / "weather_train.csv")
# weather_test = pd.read_csv(utils.DATA_DIR / "weather_test.csv")

try:

    experiment_name = datetime.strftime(datetime.now(), '%Y%m%d%H%M%S')

    if args.debug:
        result_dir = Path(utils.RESULTS_BASE_DIR) / \
            ('debug-' + experiment_name)
        N_FOLDS = 2
    else:
        result_dir = Path(utils.RESULTS_BASE_DIR) / experiment_name
        slack.notify_start(experiment_name)
    os.mkdir(result_dir)

    logger = mylogger.get_mylogger(filename=result_dir / 'log')

    logger.debug(f'created: {result_dir}')

    logger.debug('loading data ...')

    train = dataset.AshraeDataset(mode='train', debug=args.debug)
    test = dataset.AshraeDataset(mode='test')

    # preprocessing
    bad_rows = preprocessing.find_bad_rows(train.merged_df)
    # print(bad_rows)
    train.merged_df = train.merged_df.drop(index=bad_rows)
    # extract feature
    logger.debug('extracting features ...')
    # x = train.merged_df.copy()
    # test_x = test.merged_df.copy()
    x = pd.concat([train.merged_df.copy(),
                   features.time_feature(train.merged_df)],
                  axis=1)
    test_x = pd.concat([test.merged_df.copy(),
                        features.time_feature(test.merged_df)],
                       axis=1)

    le = LabelEncoder()
    le.fit(utils.PRIMARY_USE)
    x['primary_use'] = le.transform(x['primary_use'])
    test_x['primary_use'] = le.transform(test_x['primary_use'])

    x['meter_reading'] = np.log1p(x['meter_reading'])

    categorical_features = [
        'building_id_0',
        'building_id_1',
        'site_id',
        'meter',
        'primary_use',
        'had_air_temperature',
        'had_cloud_coverage',
        'had_dew_temperature',
        'had_precip_depth_1_hr',
        'had_sea_level_pressure',
        'had_wind_direction',
        'had_wind_speed'
    ]

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
        'verbose': -1,
        'device': 'gpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0

    }

    # reindex for cv
    x.index = range(len(x))

    # because of high cardinality
    x = preprocessing.split_building_id(x)
    test_x = preprocessing.split_building_id(test_x)

    logger.debug(x.columns)
    logger.debug(x.shape)

    if args.debug:
        folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1001)
        cv_indices = list(folds.split(x))
        # print(cv_indices)
    else:
        cv_indices = utils.get_cv_index(x)
        # print(cv_indices)

    x = x.drop(['timestamp'], axis=1)
    # print(x.dtypes)
    # print(x['had_air_temperature'])
    test_x = test_x.drop(['timestamp'], axis=1)

    # logger.debug(len(x))
    y_preds = np.zeros(len(x))
    for n_fold, (train_idx, val_idx) in enumerate(cv_indices):
        print(min(train_idx), max(train_idx))
        print(min(val_idx), max(val_idx))
        train_x, val_x = x.iloc[train_idx], x.iloc[val_idx]
        train_y, val_y = train_x['meter_reading'], val_x['meter_reading']
        train_x = train_x.drop('meter_reading', axis=1)
        val_x = val_x.drop('meter_reading', axis=1)

        log_evaluater = mycallbacks.log_evaluation(logger=logger, period=100)
        callbacks = [log_evaluater]
        model = LGBMRegressor(**default_param)
        model.fit(train_x, train_y, eval_set=[(train_x, train_y), (val_x, val_y)],
                  eval_metric='rmse', verbose=100, early_stopping_rounds=200,
                  categorical_feature=categorical_features,
                  callbacks=callbacks)
        y_preds[val_idx] = model.predict(
            val_x, num_iteration=model.best_iteration_)
        model_path = result_dir / f'lgbm_fold{n_fold}.pkl'
        utils.dump_pickle(model, model_path)

    y_preds = np.where(y_preds < 0, 0, y_preds)
    val_score = np.sqrt(mean_squared_error(y_preds, x['meter_reading']))
    logger.debug(val_score)

    # fit on full train data
    logger.debug('fitting on full train data')
    log_evaluater = mycallbacks.log_evaluation(logger=logger, period=100)
    callbacks = [log_evaluater]
    model = LGBMRegressor(**default_param)
    model.fit(x.drop(['meter_reading'], axis=1), x['meter_reading'],
              eval_metric='rmse', verbose=100, callbacks=callbacks)
    model_path = result_dir / f'lgbm_all.pkl'
    utils.dump_pickle(model, model_path)

    # predict test data
    # print(set(x.columns) - set(test_x.columns))
    # print(set(test_x.columns) - set(x.columns))
    logger.debug(test_x.columns)
    logger.debug(test_x.shape)
    logger.debug(set(x.columns) - set(test_x.columns))
    logger.debug(set(test_x.columns) - set(x.columns))
    test_preds = np.zeros(len(test_x))
    model = utils.load_pickle(result_dir / 'lgbm_all.pkl')
    test_preds = model.predict(test_x.drop(['row_id'], axis=1),
                               num_iteration=model.best_iteration_)
    # for i in tqdm(range(N_FOLDS)):
    #     model = utils.load_pickle(result_dir / f'lgbm_fold{i}.pkl')
    #     test_preds += model.predict(test_x.drop(['row_id'], axis=1),
    #                                 num_iteration=model.best_iteration_)
    # test_preds /= 5

    sample_submission = utils.load_pickle(
        utils.DATA_DIR / 'sample_submission.pkl')
    sample_submission['meter_reading'] = np.expm1(test_preds)
    sample_submission.loc[sample_submission['meter_reading']
                          < 0, 'meter_reading'] = 0
    submit_save_path = result_dir / f'submission_{val_score:.5f}.csv'
    sample_submission.to_csv(submit_save_path, index=False)
    logger.debug(f'save to {submit_save_path}')
    if not args.debug:
        slack.notify_finish(experiment_name, val_score)

except Exception as e:
    logger.exception(e)
    if not args.debug:
        slack.notify_fail(experiment_name, e.__class__.__name__, str(e))
