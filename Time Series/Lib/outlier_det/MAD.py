from Lib import kll


class MAD:
    def __init__(self, th=3.0):
        self.th = th
        self.deviation = []
        self.mad = 1.0
        self.dev_quant = kll.KLL(256)

    def add_sample(self, median, x):
        outlier = False
        dev = x - median
        self.dev_quant.update(dev)
        self.mad = self.compute_quantile()
        if self.mad != 0 and (abs((x - median)) / self.mad) > self.th:
            return True
        return outlier

    # def mad(self):

    def compute_quantile(self):
        a = list(zip(*self.dev_quant.cdf()))
        b = list(filter(lambda i: i > 0.5, a[1]))[0]
        idx = a[1].index(b)
        return a[0][idx]
