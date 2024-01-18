import pandas as pd
import numpy as np

class Interpolation:

    def __init__(self):
        self.added_values = []

    def interpolate(self, df, ml_method):
        df_int = df.copy()
        df_int = df_int.fillna(value=np.nan)
        if ml_method == 'regression':
            df_int[df.station == ' 0'] = df_int[df.station == ' 0'].interpolate(method='linear')
            df_int[df.station == ' 1'] = df_int[df.station == ' 1'].interpolate(method='linear')
            df_int[df.station == ' 2'] = df_int[df.station == ' 2'].interpolate(method='linear')
        elif ml_method == 'classification':
            df_int = df_int.interpolate(method='linear')
        df_int = df_int.bfill()
        df_int = df_int.ffill()

        interpolated_mask = df.isnull() & ~df_int.isnull()

        interpolated_values = df_int.values[interpolated_mask]

        self.added_values.extend(interpolated_values.tolist())

        return df_int