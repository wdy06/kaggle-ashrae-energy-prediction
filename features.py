import pandas as pd
import numpy as np
import math


def time_feature(df):
    result_df = pd.DataFrame()
    result_df['month'] = df['timestamp'].dt.month.astype(np.int)
    result_df['weekofyear'] = df['timestamp'].dt.weekofyear.astype(np.int8)
    result_df['dayofyear'] = df['timestamp'].dt.dayofyear.astype(np.int16)
    result_df['hour'] = df['timestamp'].dt.hour.astype(np.int8)
    result_df['dayofweek'] = df['timestamp'].dt.dayofweek.astype(np.int8)
    result_df['day_month'] = df['timestamp'].dt.day.astype(np.int8)
    result_df['week_month'] = df['timestamp'].dt.day / 7
    result_df['week_month'] = result_df['week_month'].apply(
        lambda x: math.ceil(x)).astype(np.int8)
    return result_df
