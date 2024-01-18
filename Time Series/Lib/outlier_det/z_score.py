import numpy as np


class z_score:

    def __init__(self, types):
        self.mean = np.zeros(len(types))
        self.std_dev = np.zeros(len(types))
        self.count = np.zeros(len(types))
        self.types = types

    def add_sample(self, row):
        for col in range(len(self.types)):
            if row[col] is not None:
                self.count[col] += 1
        self.update_std_dev(row)
        self.update_mean(row)
        o = self.detect_outliers(row)
        return o

    def update_mean(self, row):
        for col in range(len(self.types)):
            if (self.types[col] == 'int' or self.types[col] == 'float') and row[col] is not None:
                self.mean[col] = (self.mean[col] * (self.count[col] - 1) + float(row[col])) / self.count[col]

    def update_std_dev(self, row):
        for col in range(len(self.types)):
            if (self.types[col] == 'int' or self.types[col] == 'float') and row[col] is not None:
                if self.count[col] < 2:
                    a = 0
                else:
                    a = (self.count[col] - 2) / (self.count[col] - 1)
                self.std_dev[col] = a * self.std_dev[col] + (1 / self.count[col]) * ((float(row[col]) - self.mean[col]) ** 2)

    def detect_outliers(self, row):
        o = False
        for col in range(len(self.types)):
            if (self.types[col] == 'int' or self.types[col] == 'float') and row[col] is not None:
                z = abs(float(row[col]) - self.mean[col]) / (self.std_dev[col])
                if z > 2.5:
                    o = True
        return o


    def print_mean_std_dev(self):
        print("MEAN: ",self.mean)
        print("STD DEV: ",self.std_dev)