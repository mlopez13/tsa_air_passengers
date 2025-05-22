import numpy as np
import pandas as pd

from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from modules import LSTMNet


class Model:
    def __init__(self):
        pass

    def get_metrics(self, val, pred):
        mae = mean_absolute_error(val["Passengers"], pred)
        rmse = np.sqrt(mean_squared_error(val["Passengers"], pred))
        metrics = {
            "MAE": mae,
            "RMSE": rmse,
        }
        return metrics


class Naive(Model):
    def __init__(self, train, val):
        self.train = train
        self.val = val

    def train_model(self):
        decompose = seasonal_decompose(
            self.train["Passengers"], model="additive", period=12, two_sided=False
        )
        y_trend = decompose.trend.dropna().values
        self.y_seasonal = decompose.seasonal.dropna().values
        x_trend = self.train[["Month"]][-len(y_trend) :]
        self.model = LinearRegression().fit(x_trend, y_trend)

    def predict(self):
        y_pred_trend = self.model.predict(self.val[["Month"]])
        pred = y_pred_trend + self.y_seasonal[-len(self.val) :]
        return pred


class HoltWinters(Model):
    def __init__(self, train, val):
        self.train = train.set_index("Date")
        self.train.index.freq = "MS"
        self.val = val.set_index("Date")
        self.val.index.freq = "MS"

    def train_model(self):
        self.model = ExponentialSmoothing(
            endog=self.train["Passengers"],
            trend="add",
            seasonal="add",
            seasonal_periods=12,
        )
        self.model = self.model.fit()

    def predict(self):
        pred = self.model.forecast(len(self.val))
        return pred


class SARIMA(Model):
    def __init__(self, train, val, order, seasonal_order):
        super().__init__()
        self.train = train.set_index("Date")
        self.train.index.freq = "MS"
        self.val = val.set_index("Date")
        self.val.index.freq = "MS"
        self.order = order
        self.seasonal_order = seasonal_order

    def train_model(self):
        self.model = ARIMA(
            self.train["Passengers"],
            order=self.order,
            seasonal_order=self.seasonal_order,
        )
        self.model = self.model.fit()

    def predict(self):
        pred = self.model.predict(start=self.val.index[0], end=self.val.index[-1])
        return pred.values


class LSTM(Model):
    def __init__(self, train, val, n_epochs=2500):

        super().__init__()
        self.n_months_linreg = 36
        self.look_back = 12
        self.recurrent_size = 50
        self.n_layers = 1
        self.batch_size = 8
        self.n_epochs = n_epochs

        self.train_res_scal, self.val_res_scal = self.prepare_data(train, val)

    def prepare_data(self, train, val):

        train_subset = train[-self.n_months_linreg :]

        lin_reg = LinearRegression()
        lin_reg = lin_reg.fit(train_subset[["Month"]], train_subset["Passengers"])
        train_fit = lin_reg.predict(train[["Month"]])
        self.val_fit = lin_reg.predict(val[["Month"]])

        train["res"] = train["Passengers"] - train_fit
        val["res"] = val["Passengers"] - self.val_fit

        # Normalize the data
        self.scaler = MinMaxScaler()
        train["res_scal"] = self.scaler.fit_transform(train[["res"]])
        val["res_scal"] = self.scaler.fit_transform(val[["res"]])

        train["res_scal"] = train["res_scal"].astype("float32")
        val["res_scal"] = val["res_scal"].astype("float32")

        return train["res_scal"].values, val["res_scal"].values

    def create_dataset(self, data, look_back):
        X, y = [], []
        for i in range(len(data) - look_back):
            feature = data[i : i + look_back].reshape(-1, 1)
            target = data[i + 1 : i + look_back + 1]
            X.append(feature)
            y.append(target)
        X = np.array(X)
        y = np.array(y)
        return torch.tensor(X), torch.tensor(y)

    def deep_learning_loop(self, train_dataloader, is_dropout=False):

        self.model = LSTMNet(
            dim_input=1,
            dim_recurrent=self.recurrent_size,
            num_layers=self.n_layers,
            dim_output=1,
            is_dropout=is_dropout,
        )

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters())

        for epoch in range(1, self.n_epochs + 1):
            # Set model to train mode
            self.model.train()
            for features, targets in train_dataloader:
                outputs = self.model(features)[:, :, 0]
                loss = criterion(outputs, targets)
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def train_model(self):
        self.X_train, self.y_train = self.create_dataset(
            data=self.train_res_scal, look_back=self.look_back
        )
        train_dataloader = DataLoader(
            dataset=TensorDataset(self.X_train, self.y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.deep_learning_loop(train_dataloader=train_dataloader, is_dropout=True)

    def predict(self):
        pred = torch.zeros(len(self.val_res_scal))
        last_values = self.X_train[-1]

        for i in range(self.look_back):
            self.model.eval()
            with torch.no_grad():
                new_pred = self.model(last_values)[-1]
            last_values = torch.vstack((last_values[1:], new_pred))
            pred[i] = new_pred

        for i in range(len(self.val_res_scal) - self.look_back):
            last_values = pred[i : i + self.look_back].reshape(-1, 1)
            self.model.eval()
            with torch.no_grad():
                new_pred = self.model(last_values)[-1]
            pred[i + self.look_back] = new_pred

        # Reverse scaling, and add LinReg prediction
        pred = (
            self.scaler.inverse_transform(pred.reshape(-1, 1)).reshape(1, -1)[0]
            + self.val_fit
        )

        return pred
