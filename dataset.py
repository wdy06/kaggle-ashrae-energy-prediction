import pandas as pd
import numpy as np

import utils
import preprocessing


class AshraeDataset():
    def __init__(self, mode='train', debug=False):
        self.mode = mode
        self.debug = debug

        self.building_meta = utils.load_pickle(
            utils.DATA_DIR / 'building_metadata.pkl')
        self.main_df = utils.load_pickle(utils.DATA_DIR / f'{mode}.pkl')
        if self.debug and self.mode == 'train':
            self.main_df = self.main_df[:2000]
        self.weather = utils.load_pickle(
            utils.DATA_DIR / f'weather_{mode}.pkl')

        self.main_df['timestamp'] = pd.to_datetime(self.main_df['timestamp'])
        self.weather['timestamp'] = pd.to_datetime(self.weather['timestamp'])

        # self.main_df = preprocessing.fix_timestamp_to_hour_standard(
        #     self.main_df)
        # self.weather = preprocessing.fix_timestamp_to_hour_standard(
        #     self.weather)

        # self.weather = preprocessing.interpolate_weather_data(
        #     self.weather, mode=self.mode)
        # self.merged_df = pd.merge(self.main_df, self.building_meta,
        #                           on='building_id', how='left')
        # self.merged_df = pd.merge(self.merged_df, self.weather,
        #                           on=['site_id', 'timestamp'], how='left')

        self.merged_df = utils.load_pickle(
            utils.DATA_DIR / f'clean_merged_{mode}.pkl')
        if self.debug:
            self.merged_df = self.merged_df[:2000]

    def __len__(self):
        return len(self.main_df)
