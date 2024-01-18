import Lib.profiling_lib as pro
import numpy as np

class r_mean:

    def __init__(self,types):
        self.mean = []
        self.count = []
        self.predicted_values = []
        for i in range(3):
            self.mean.append(np.zeros(len(types)))
            self.count.append(np.zeros(len(types)))
        self.types = types

    def update(self, row):
        station = int(row[13])
        for col in range(len(row)):
            if row[col] is not None:
                self.count[station][col] += 1
        self.mean[station] = pro.c_mean(self.mean[station], self.count[station], row, self.types)

    def impute(self, row):
        station = int(row[13])
        for col in range(len(row)):
            if row[col] is None:
                row[col] = self.mean[station][col]
                self.predicted_values.append(row[col])
        return row