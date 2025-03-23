import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression


class Model():
    def __init__(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass


class Naive(Model):
    def __init__(self):
        super().__init__()

    def train(self, train):
        decompose = seasonal_decompose(train, model='additive', period=12, two_sided=False)
        y_trend = decompose.trend.dropna().values
        self.y_seasonal = decompose.seasonal.dropna().values

        self.x = np.arange(len(y_trend)).reshape(-1, 1)

        self.reg = LinearRegression().fit(self.x, y_trend)

    def predict(self, val):
        x_pred = np.arange(self.x[-1][0] + 1, self.x[-1][0] + len(val) + 1).reshape(-1, 1)
        y_pred_trend = self.reg.predict(x_pred)
        y_pred_trend = pd.Series(data=y_pred_trend, index=val.index)
        self.y_pred = y_pred_trend + self.y_seasonal[- len(val):]
        return self.y_pred

class MA(Model):
    def __init__(self):
        super().__init__()

class AR(Model):
    def __init__(self):
        super().__init__()
