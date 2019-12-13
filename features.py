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


def holiday_feature(df):
    org_df = df.copy()
    result_df = pd.DataFrame()
    # result_df['building_id'] = org_df['building_id']
    # result_df['timestamp'] = org_df['timestamp']
    result_df['holiday'] = org_df['timestamp'].dt.dayofweek.map(
        lambda x: True if x in [5, 6] else False)
    return result_df


def lag_shift_feature(df):
    result_df = df.copy()
    result_df = result_df.groupby(['building_id', 'meter']).apply(
        lambda group: group.sort_values(by='timestamp', axis=0))
    features = ['air_temperature', 'dew_temperature',
                'cloud_coverage', 'precip_depth_1_hr', 'sea_level_pressure', 'wind_direction', 'wind_speed']
    period_list = [1, 2, 3, 4, 5, 6]
    shift_feature_list = []
    for feature_col in features:
        for period in period_list:
            shift_feature_name = f'lag_{period}day_{feature_col}'
            result_df[shift_feature_name] = _lag_shift(
                result_df, feature_col, period)
            shift_feature_list.append(shift_feature_name)

    result_df = result_df[['building_id', 'meter',
                           'timestamp'] + shift_feature_list]
    result_df.index = range(len(result_df))
    return result_df


def _lag_shift(df, column, periods):
    return df[column].groupby(['building_id', 'meter']).shift(periods=periods)


def aggregate_weather_feature(df):
    org_df = df.copy()
    aggregations = {
        'air_temperature': {
            'air_temp_mean': 'mean',
            'air_temp_median': 'median',
            'air_temp_min': 'min',
            'air_temp_max': 'max',
            'air_temp_std': 'std'
        },
        'cloud_coverage': {
            'cloud_coverage_mean': 'mean',
            'cloud_coverage_median': 'median',
            'cloud_coverage_min': 'min',
            'cloud_coverage_max': 'max',
            'cloud_coverage_std': 'std'
        },
        'dew_temperature': {
            'dew_temp_mean': 'mean',
            'dew_temp_median': 'median',
            'dew_temp_min': 'min',
            'dew_temp_max': 'max',
            'dew_temp_std': 'std'
        },
        'precip_depth_1_hr': {
            'precip_depth_1_hr_mean': 'mean',
            'precip_depth_1_hr_median': 'median',
            'precip_depth_1_hr_min': 'min',
            'precip_depth_1_hr_max': 'max',
            'precip_depth_1_hr_std': 'std'
        },
        'sea_level_pressure': {
            'sea_level_pressure_mean': 'mean',
            'sea_level_pressure_median': 'median',
            'sea_level_pressure_min': 'min',
            'sea_level_pressure_max': 'max',
            'sea_level_pressure_std': 'std'
        },
        'wind_direction': {
            'wind_direction_mean': 'mean',
            'wind_direction_median': 'median',
            'wind_direction_min': 'min',
            'wind_direction_max': 'max',
            'wind_direction_std': 'std'
        },
        'wind_speed': {
            'wind_speed_mean': 'mean',
            'wind_speed_median': 'median',
            'wind_speed_min': 'min',
            'wind_speed_max': 'max',
            'wind_speed_std': 'std'
        },
    }

    aggregated = org_df.groupby(
        ['site_id', 'meter', 'month']).agg(aggregations)
    aggregated.columns = aggregated.columns.get_level_values(1)
    return aggregated


def aggregate_meter_reading(df):
    org_df = df.copy()
    aggregations = {
        'meter_reading': {
            'meter_reading_mean': 'mean',
            'meter_reading_median': 'median',
            'meter_reading_min': 'min',
            'meter_reading_max': 'max',
            'meter_reading_std': 'std',
        }
    }
    aggregated = org_df.groupby(
        ['building_id', 'meter', 'month']).agg(aggregations)
    aggregated.columns = aggregated.columns.get_level_values(1)
    return aggregated
