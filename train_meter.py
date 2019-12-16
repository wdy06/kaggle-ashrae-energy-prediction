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
METER_TYPE_LIST = [0, 1, 2, 3]


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
    train.merged_df = train.merged_df.drop(index=bad_rows)
    # extract feature
    logger.debug('extracting features ...')
    # time feature
    x = pd.merge(train.merged_df.copy(),
                 utils.load_pickle(utils.FEATURE_DIR /
                                   'train_time_feature.pkl'),
                 on=['building_id', 'meter', 'timestamp'],
                 how='left')
    test_x = pd.merge(test.merged_df.copy(),
                      utils.load_pickle(utils.FEATURE_DIR /
                                        'test_time_feature.pkl'),
                      on=['building_id', 'meter', 'timestamp'],
                      how='left')
    # holiday feature
    x = pd.merge(x,
                 utils.load_pickle(utils.FEATURE_DIR /
                                   'train_holiday_feature.pkl'),
                 on=['building_id', 'meter', 'timestamp'],
                 how='left')
    test_x = pd.merge(test_x,
                      utils.load_pickle(utils.FEATURE_DIR /
                                        'test_holiday_feature.pkl'),
                      on=['building_id', 'meter', 'timestamp'],
                      how='left')

    # lag shfit feature
    x = pd.merge(x, utils.load_pickle(utils.FEATURE_DIR / 'train_lag_shift_feature.pkl'),
                 on=['building_id', 'meter', 'timestamp'],
                 how='left')
    test_x = pd.merge(test_x, utils.load_pickle(utils.FEATURE_DIR / 'test_lag_shift_feature.pkl'),
                      on=['building_id', 'meter', 'timestamp'],
                      how='left')

    # aggregation feature
    # x = x.join(utils.load_pickle(utils.FEATURE_DIR / 'train_aggregate_weather_feature.pkl'),
    #            on=['site_id', 'meter', 'month'])
    # test_x = test_x.join(utils.load_pickle(utils.FEATURE_DIR / 'test_aggregate_weather_feature.pkl'),
    #                      on=['site_id', 'meter', 'month'])
    # meter aggregation
    # meter_aggregated = utils.load_pickle(
    #     utils.FEATURE_DIR / 'train_aggregate_meter_reading_feature.pkl')
    # x = x.join(meter_aggregated, on=['building_id', 'meter', 'month'])
    # test_x = test_x.join(meter_aggregated, on=[
    #                      'building_id', 'meter', 'month'])

    x = preprocessing.log_square_feet(x)
    test_x = preprocessing.log_square_feet(test_x)
    le = LabelEncoder()
    le.fit(utils.PRIMARY_USE)
    x['primary_use'] = le.transform(x['primary_use'])
    test_x['primary_use'] = le.transform(test_x['primary_use'])

    x['meter_reading'] = np.log1p(x['meter_reading'])

    categorical_features = [
        'building_id_0',
        'building_id_1',
        'site_id',
        # 'meter',
        'primary_use',
        'holiday',
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
        'learning_rate': 0.1,
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
        'gpu_device_id': 0,
        'random_state': 2019,

    }

    # reindex for cv
    x.index = range(len(x))
    test_x.index = range(len(test_x))

    # add leak score column
    test_x = utils.leak_validation(test_x)

    # because of high cardinality
    x = preprocessing.split_building_id(x)
    test_x = preprocessing.split_building_id(test_x)

    logger.debug(x.columns)
    logger.debug(x.shape)

    if args.debug:
        folds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=1001)
        cv_indices = list(folds.split(x))
    else:
        cv_indices = utils.get_cv_index(x)

    x = x.drop(['timestamp'], axis=1)
    test_x = test_x.drop(['timestamp'], axis=1)

    # dump features
    x = utils.compress_dataframe(x)
    test_x = utils.compress_dataframe(test_x)
    utils.dump_pickle(x, result_dir / 'x.pkl')
    utils.dump_pickle(test_x, result_dir / 'test_x.pkl')

    y_preds = np.zeros(len(x))
    for n_fold, (train_idx, val_idx) in enumerate(cv_indices):
        train_x, val_x = x.iloc[train_idx], x.iloc[val_idx]
        train_y, val_y = train_x['meter_reading'], val_x['meter_reading']
        train_x = train_x.drop('meter_reading', axis=1)
        val_x = val_x.drop('meter_reading', axis=1)

        for meter_type in METER_TYPE_LIST:
            train_x_meter = train_x[train_x.meter == meter_type]
            train_y_meter = train_y[train_x.meter == meter_type]
            val_x_meter = val_x[val_x.meter == meter_type]
            val_y_meter = val_y[val_x.meter == meter_type]
            val_meter_index = val_x_meter.index
            train_x_meter.drop(['meter'], axis=1, inplace=True)
            val_x_meter.drop(['meter'], axis=1, inplace=True)

            log_evaluater = mycallbacks.log_evaluation(
                logger=logger, period=100)
            callbacks = [log_evaluater]
            model = LGBMRegressor(**default_param)
            model.fit(train_x_meter, train_y_meter, eval_set=[(train_x_meter, train_y_meter), (val_x_meter, val_y_meter)],
                      eval_metric='rmse', verbose=100, early_stopping_rounds=100,
                      categorical_feature=categorical_features,
                      callbacks=callbacks)
            y_preds[val_meter_index] = model.predict(
                val_x_meter, num_iteration=model.best_iteration_)
            model_path = result_dir / \
                f'lgbm_meter{meter_type}_fold{n_fold}.pkl'
            utils.dump_pickle(model, model_path)

    y_preds = np.where(y_preds < 0, 0, y_preds)
    val_score = np.sqrt(mean_squared_error(y_preds, x['meter_reading']))
    logger.debug(f'val score: {val_score}')

    # fit on full train data
    logger.debug('fitting on full train data')
    log_evaluater = mycallbacks.log_evaluation(logger=logger, period=100)
    callbacks = [log_evaluater]
    model = LGBMRegressor(**default_param)
    for meter_type in METER_TYPE_LIST:
        x_meter = x[x.meter == meter_type]
        model.fit(x_meter.drop(['meter_reading', 'meter'], axis=1), x_meter['meter_reading'],
                  eval_metric='rmse', verbose=100, callbacks=callbacks)
        model_path = result_dir / f'lgbm_meter{meter_type}_all.pkl'
        utils.dump_pickle(model, model_path)
        utils.save_feature_importance(model, x_meter.drop(['meter_reading', 'meter'], axis=1).columns,
                                      result_dir / f'feature_importance_meter{meter_type}.png')

    logger.debug(test_x.columns)
    logger.debug(test_x.shape)
    logger.debug(set(x.columns) - set(test_x.columns))
    logger.debug(set(test_x.columns) - set(x.columns))
    test_preds = np.zeros(len(test_x))
    for meter_type in METER_TYPE_LIST:
        model = utils.load_pickle(
            result_dir / f'lgbm_meter{meter_type}_all.pkl')
        test_x_meter = test_x[test_x.meter == meter_type]
        test_preds[test_x_meter.index] = model.predict(test_x_meter.drop(
            ['row_id', 'leak_score', 'meter'], axis=1),
            num_iteration=model.best_iteration_)
    test_x['test_preds'] = test_preds

    # calculate leak validation score
    leak_val_df = test_x[~test_x.leak_score.isnull()][[
        'leak_score', 'test_preds']]
    leak_val_score = np.sqrt(mean_squared_error(
        leak_val_df.leak_score, leak_val_df.test_preds))
    logger.debug(f'leak val score: {leak_val_score}')
    sample_submission = utils.load_pickle(
        utils.DATA_DIR / 'sample_submission.pkl')
    sample_submission['meter_reading'] = np.expm1(test_x['test_preds'])
    sample_submission.loc[sample_submission['meter_reading']
                          < 0, 'meter_reading'] = 0
    submit_save_path = result_dir / f'submission_{val_score:.5f}.csv'
    sample_submission.to_csv(submit_save_path, index=False)
    logger.debug(f'save to {submit_save_path}')
    if not args.debug:
        slack.notify_finish(experiment_name, val_score)
        slack.notify_finish(experiment_name, leak_val_score)

except Exception as e:
    logger.exception(e)
    if not args.debug:
        slack.notify_fail(experiment_name, e.__class__.__name__, str(e))
