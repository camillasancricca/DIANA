import arimafd as oa
import numpy as np

class ARIMA:
    def __init__(self, columns):
        self.models = []
        self.columns = columns[1:]
        for column in columns:
            if column == 'int' or column == 'float':
                self.models.append(oa.Arima_anomaly_detection())

    def fit(self, df):
        for i in range(len(self.columns)):
            if self.columns[i] == 'int' or self.columns[i] == 'float':
                i_column = df.iloc[:,i].to_frame()
                self.models[i].fit(i_column)

    def compute_sample(self, df, count, window):
        outliers = np.array([])
        for i in range(len(self.columns)):
            if self.columns[i] == 'int' or self.columns[i] == 'float':
                i_column = df.iloc[:, i].to_frame()
                n_outliers = self.models[i].predict(i_column)
                n_outliers = n_outliers[n_outliers==1].index.values
                n_outliers = df.loc[n_outliers]["count"]
                outliers = np.union1d(outliers, n_outliers)
        self.fit(df)
        return outliers