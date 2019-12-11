import utils
import pandas as pd
import numpy as np
from datetime import timedelta
from tqdm import tqdm

tqdm.pandas()


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


def split_building_id(df):
    df['building_id'] = df['building_id'].map(lambda x: str(x).zfill(4))
    df['building_id_0'] = df['building_id'].map(lambda x: int(x[:2]))
    df['building_id_1'] = df['building_id'].map(lambda x: int(x[2:]))
    df.drop(['building_id'], axis=1, inplace=True)
    return df


def fix_timestamp_to_hour_standard(df):
    df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")
                    ).dt.total_seconds() // 3600
    return df


def reverse_timestamp(df):
    def f(x): return pd.to_datetime("2016-01-01") + timedelta(hours=x)
    df.timestamp = df.timestamp.progress_map(f)


def interpolate_weather_data(df, mode, fix_timestamps=True, interpolate_na=True, add_na_indicators=True):
    reindex_range = None
    if mode == 'train':
        reindex_range = range(8784)
    elif mode == 'test':
        reindex_range = range(8784, 26304)
    # df.timestamp = (df.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
    if fix_timestamps:
        GMT_offset_map = {site: offset for site,
                          offset in enumerate(utils.SITE_GMT_OFFSET)}
        df.timestamp = df.timestamp + df.site_id.map(GMT_offset_map)
    if interpolate_na:
        site_dfs = []
        for site_id in df.site_id.unique():
            # Make sure that we include all possible hours so that we can interpolate evenly
            site_df = df[df.site_id == site_id].set_index(
                "timestamp").reindex(reindex_range)
            site_df.site_id = site_id
            for col in [c for c in site_df.columns if c not in ["site_id"]]:
                if add_na_indicators:
                    site_df[f"had_{col}"] = ~site_df[col].isna()
                site_df[col] = site_df[col].interpolate(
                    limit_direction='both', method='linear')
                # Some sites are completely missing some columns, so use this fallback
                site_df[col] = site_df[col].fillna(df[col].median())
            site_dfs.append(site_df)

        # make timestamp back into a regular column
        df = pd.concat(site_dfs).reset_index()
    elif add_na_indicators:
        for col in df.columns:
            if df[col].isna().any():
                df[f"had_{col}"] = ~df[col].isna()
    return df


def make_is_bad_zero(Xy_subset, min_interval=48, summer_start=3000, summer_end=7500):
    """Helper routine for 'find_bad_zeros'.

    This operates upon a single dataframe produced by 'groupby'. We expect an
    additional column 'meter_id' which is a duplicate of 'meter' because groupby
    eliminates the original one."""
    meter = Xy_subset.meter_id.iloc[0]
    is_zero = Xy_subset.meter_reading == 0
    if meter == 0:
        # Electrical meters should never be zero. Keep all zero-readings in this table so that
        # they will all be dropped in the train set.
        return is_zero

    transitions = (is_zero != is_zero.shift(1))
    all_sequence_ids = transitions.cumsum()
    ids = all_sequence_ids[is_zero].rename("ids")
    if meter in [2, 3]:
        # It's normal for steam and hotwater to be turned off during the summer
        keep = set(ids[(Xy_subset.tmp_timestamp < summer_start) |
                       (Xy_subset.tmp_timestamp > summer_end)].unique())
        is_bad = ids.isin(keep) & (ids.map(ids.value_counts()) >= min_interval)
    elif meter == 1:
        time_ids = ids.to_frame().join(Xy_subset.tmp_timestamp).set_index("tmp_timestamp").ids
        is_bad = ids.map(ids.value_counts()) >= min_interval

        # Cold water may be turned off during the winter
        jan_id = time_ids.get(0, False)
        dec_id = time_ids.get(8283, False)
        if (jan_id and dec_id and jan_id == time_ids.get(500, False) and
                dec_id == time_ids.get(8783, False)):
            is_bad = is_bad & (~(ids.isin(set([jan_id, dec_id]))))
    else:
        raise Exception(f"Unexpected meter type: {meter}")

    result = is_zero.copy()
    result.update(is_bad)
    return result


def find_bad_zeros(X):
    """Returns an Index object containing only the rows which should be deleted."""
    X_tmp = X.assign(meter_id=X.meter)
    is_bad_zero = X_tmp.groupby(
        ["building_id", "meter"]).apply(make_is_bad_zero)
    return is_bad_zero[is_bad_zero].index.droplevel([0, 1])


def find_bad_sitezero(X):
    """Returns indices of bad rows from the early days of Site 0 (UCF)."""
    return X[(X.tmp_timestamp < 3378) & (X.site_id == 0) & (X.meter == 0)].index


def find_bad_building1099(X):
    """Returns indices of bad rows (with absurdly high readings) from building 1099."""
    return X[(X.building_id == 1099) & (X.meter == 2) & (X.meter_reading > 3e4)].index


def find_bad_rows(X):
    X["tmp_timestamp"] = (
        X.timestamp - pd.to_datetime("2016-01-01")).dt.total_seconds() // 3600
    bad_row_index = find_bad_zeros(X).union(
        find_bad_sitezero(X)).union(find_bad_building1099(X))
    X.drop(["tmp_timestamp"], axis=1, inplace=True)
    return bad_row_index


def log_square_feet(df):
    df['square_feet'] = np.log(df['square_feet'])
    df.rename(columns={'square_feet': 'log_square_feet'}, inplace=True)
    return df
