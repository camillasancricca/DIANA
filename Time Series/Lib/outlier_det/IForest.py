from pyod.models.iforest import IForest
import numpy as np

class isoforest:
    def __init__(self):
        self.model = IForest(random_state=42)

    def fit(self,data):
        data_c = data.copy()
        data_c = data_c.fillna(0)
        if 'date_time' in data_c.columns:
            data_c.drop(['date_time', 'arrive_time'], axis=1, inplace=True)
        else :
            data_c.drop(['arrive_time'],axis=1, inplace=True)
        self.model = self.model.fit(data_c)

    def predict(self,data,count,window,flag=False):
        data_c = data.copy()
        data_c = data_c.fillna(0)
        if 'date_time' in data_c.columns:
            data_c.drop(['date_time', 'arrive_time'], axis=1, inplace=True)
        else:
            data_c.drop(['arrive_time'], axis=1, inplace=True)
        data_c = data_c.to_numpy()
        data_c = np.nan_to_num(data_c)
        outliers = self.model.predict(data_c)
        self.model = self.model.fit(data_c)
        outliers = np.where(outliers == 1)[0]
        if flag == False:
            outliers += (count - window)
        return outliers

