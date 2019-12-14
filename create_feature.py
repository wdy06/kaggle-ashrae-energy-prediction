import numpy as np
import pandas as pd

from dataset import AshraeDataset
import features
import utils

func_map = {
    'time_feature': features.time_feature,
    'holiday_feature': features.holiday_feature,
    'lag_shift_feature': features.lag_shift_feature,
    'aggregate_weather_feature': features.aggregate_weather_feature,
    'aggregate_meter_reading': features.aggregate_meter_reading
}


def create_feature(feature_name, mode, output_path):
    data = AshraeDataset(mode=mode)
    feat_func = func_map[feature_name]
    feat = feat_func(data.merged_df)
    utils.dump_pickle(feat, output_path)
