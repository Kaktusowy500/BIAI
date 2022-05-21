import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.autograd import Variable
import torch




class DataPreparer:

    def __init__(self):

        self.mm = MinMaxScaler()
        self.ss = StandardScaler()
        self.X_train_tensor = None
        self.X_test_tensor = None
        self.y_train_tensor = None
        self.y_test_tensor = None

    def split_and_convert_to_tensors(self, df):

        # Prepare data
        X = df.iloc[:, :-1]  # params without volume
        y = df.iloc[:, 5:6]  # volume

       
        X_ss = self.ss.fit_transform(X)
        y_mm = self.mm.fit_transform(y)

        X_train = X_ss[:200, :]
        X_test = X_ss[200:, :]

        y_train = y_mm[:200, :]
        y_test = y_mm[200:, :]

        self.X_train_tensor = Variable(torch.Tensor(X_train))
        self.X_test_tensor = Variable(torch.Tensor(X_test))

        self.y_train_tensor = Variable(torch.Tensor(y_train))
        self.y_test_tensor = Variable(torch.Tensor(y_test))

    def normalize_df_to_tensor(self, df):
        df_X_ss = self.ss.transform(df.iloc[:, :-1])  # old transformers
        df_y_mm = self.mm.transform(df.iloc[:, -1:])  # old transformers

        df_X_ss = Variable(torch.Tensor(df_X_ss))  # converting to Tensors
        df_y_mm = Variable(torch.Tensor(df_y_mm))

        return df_X_ss, df_y_mm

    def mm_inverse_transform(self, data):
        return self.mm.inverse_transform(data)  # reverse transformation

