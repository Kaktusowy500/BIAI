from matplotlib.pyplot import axis
import pandas as pd
import torch
import numpy as np

from sklearn.preprocessing import MinMaxScaler


class DataPrep:
    def __init__(self):
        self.df = None
        self.scaler_price = MinMaxScaler()
        self.scaler_volume = MinMaxScaler()
        self.is_cuda = False

    def read_and_parse_df(self, path, dt_from=None, dt_to=None):
        """read data frame from csv and extract price change"""
        df = pd.read_csv(path, index_col='Date', parse_dates=True)

        if dt_to:
            df = df.loc[:dt_to]
        if dt_from:
            df = df.loc[dt_from:]

        self.df = df
        return self.df

    def train_test(self, test_periods):
        """split df to train and test sets"""
        train = self.df[:-test_periods].values
        test = self.df[-test_periods:].values

        return train, test

    def scale_data(self, data_set):
        """Scale data 0-1 range and convert to flat tensor"""
        price = data_set[:, 1].reshape(-1, 1)
        volume =  data_set[:, 2].reshape(-1, 1)
        self.scaler_price.fit(price)
        self.scaler_volume.fit(volume)
        scaled_price = np.array(self.scaler_price.transform(price))
        scaled_volume = np.array(self.scaler_volume.transform(volume))
        scaled_data = torch.FloatTensor(np.concatenate((scaled_price, scaled_volume), axis=1))

        if self.is_cuda:
            scaled_data = scaled_data.cuda()

        return scaled_data

    def get_x_y_pairs(self, train_scaled, train_periods, prediction_periods):
        """Get x, y pairs of data for training"""
        x_train = [train_scaled[i:i + train_periods] for i in range(len(train_scaled) - train_periods - prediction_periods)]
        y_train = [train_scaled[i + train_periods:i + train_periods + prediction_periods, 0] for i in range(len(train_scaled) - train_periods - prediction_periods)]

        x_train = torch.stack(x_train)
        y_train = torch.stack(y_train)

        if self.is_cuda:
            x_train = x_train.cuda()
            y_train = y_train.cuda()

        return x_train, y_train

    def cuda(self):
        self.is_cuda = True
