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

# logger = getLogger(__name__)

parser = argparse.ArgumentParser(description='kaggle ashrae energy prediction')
parser.add_argument("--debug", help="run debug mode",
                    action="store_true")
parser.add_argument("--input", type=str, help='input feature path')
parser.add_argument("--model", type=str, help='model path')
args = parser.parse_args()

N_FOLDS = 6


print('loading data ...')
x = utils.load_pickle(args.input)

test_preds = np.zeros(len(x))
model = utils.load_pickle(args.model)
print('predicting ...')
test_preds = model.predict(x.drop(['row_id'], axis=1),
                           num_iteration=model.best_iteration_, verbose=1)
print('finished predicting ...')

sample_submission = utils.load_pickle(
    utils.DATA_DIR / 'sample_submission.pkl')
sample_submission['meter_reading'] = np.expm1(test_preds)
sample_submission.loc[sample_submission['meter_reading']
                      < 0, 'meter_reading'] = 0
# submit_save_path = result_dir / f'submission_{val_score:.5f}.csv'
submit_save_path = 'submission.csv'
sample_submission.to_csv(submit_save_path, index=False)
# logger.debug(f'save to {submit_save_path}')
# if not args.debug:
#     slack.notify_finish(experiment_name, val_score)
