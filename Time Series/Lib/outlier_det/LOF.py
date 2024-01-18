from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import numpy as np

class LOF:

    def __init__(self,n):
        self.model = LocalOutlierFactor(n_neighbors=n)
        self.scaler = StandardScaler()

    def compute(self, df, count, window, flag = False):
        #df.drop(['date_time','arrive_time'],axis=1,inplace=True)
        data = df.to_numpy()
        data = np.nan_to_num(self.normalize(data))
        outliers = self.model.fit_predict(data)
        outliers = np.where(outliers == -1)[0]
        if flag == False:
            outliers += (count-window)
        return outliers

    def normalize(self,data):
        data = self.scaler.fit_transform(data)
        return data

