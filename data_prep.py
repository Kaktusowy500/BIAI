import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


class DataPrep:
    def __init__(self):
        self.df = None
        self.scaler = MinMaxScaler()

    def read_and_parse_df(self, path):
        """read data frame from csv and extract price change"""
        df = pd.read_csv(path, index_col='Date', parse_dates=True)
        self.df = df.iloc[:, 1:2]

    def train_test(self, test_periods):
        """split df to train and test sets"""
        train = self.df[:-test_periods].values
        test = self.df[-test_periods:].values
        return train, test

    def scale_data(self, data_set):
        """Scale data 0-1 range and convert to flat tensor"""
        self.scaler.fit(data_set)
        scaled_data = self.scaler.transform(data_set)
        scaled_data = torch.FloatTensor(scaled_data)
        return scaled_data.view(-1)

    def get_x_y_pairs(self, train_scaled, train_periods, prediction_periods):
        """Get x, y pairs of data for training"""
        x_train = [train_scaled[i:i + train_periods] for i in range(len(train_scaled) - train_periods - prediction_periods)]
        y_train = [train_scaled[i + train_periods:i + train_periods + prediction_periods] for i in range(len(train_scaled) - train_periods - prediction_periods)]

        x_train = torch.stack(x_train)
        y_train = torch.stack(y_train)

        return x_train, y_train
