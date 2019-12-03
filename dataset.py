import pandas as pd
import numpy as np

import utils


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

        self.merged_df = pd.merge(self.main_df, self.building_meta,
                                  on='building_id', how='left')
        self.merged_df = pd.merge(self.merged_df, self.weather,
                                  on=['site_id', 'timestamp'], how='left')

    def __len__(self):
        return len(self.main_df)
