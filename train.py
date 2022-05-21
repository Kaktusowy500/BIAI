from lstm import CustomLSTM
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt


df = pd.read_csv('SBUX.csv', index_col='Date', parse_dates=True)

# Prepare data
X = df.iloc[:, :-1]  # params without volume
y = df.iloc[:, 5:6]  # volume

mm = MinMaxScaler()
ss = StandardScaler()
X_ss = ss.fit_transform(X)
y_mm = mm.fit_transform(y)


X_train = X_ss[:200, :]
X_test = X_ss[200:, :]

y_train = y_mm[:200, :]
y_test = y_mm[200:, :]

X_train_tensors = Variable(torch.Tensor(X_train))
X_test_tensors = Variable(torch.Tensor(X_test))

y_train_tensors = Variable(torch.Tensor(y_train))
y_test_tensors = Variable(torch.Tensor(y_test))


X_train_tensors_final = torch.reshape(X_train_tensors, (X_train_tensors.shape[0], 1, X_train_tensors.shape[1]))


X_test_tensors_final = torch.reshape(X_test_tensors, (X_test_tensors.shape[0], 1, X_test_tensors.shape[1]))

print("Training Shape", X_train_tensors_final.shape, y_train_tensors.shape)
print("Testing Shape", X_test_tensors_final.shape, y_test_tensors.shape)


num_epochs = 1000
learning_rate = 0.001

input_size = 5  # number of features
hidden_size = 2  # number of features in hidden state
num_layers = 1  # number of stacked lstm layers

num_classes = 1  # number of output classes
lstm = CustomLSTM(num_classes, input_size, hidden_size, num_layers, X_train_tensors_final.shape[1])
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    outputs = lstm.forward(X_train_tensors_final)  # forward pass
    optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

    # obtain the loss function
    loss = criterion(outputs, y_train_tensors)

    loss.backward()  # calculates the loss of the loss function

    optimizer.step()  # improve from loss, i.e backprop
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item():.5f}")


df_X_ss = ss.transform(df.iloc[:, :-1])  # old transformers
df_y_mm = mm.transform(df.iloc[:, -1:])  # old transformers

df_X_ss = Variable(torch.Tensor(df_X_ss))  # converting to Tensors
df_y_mm = Variable(torch.Tensor(df_y_mm))
# reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))

train_predict = lstm(df_X_ss)  # forward pass
data_predict = train_predict.data.numpy()  # numpy conversion
dataY_plot = df_y_mm.data.numpy()

data_predict = mm.inverse_transform(data_predict)  # reverse transformation
dataY_plot = mm.inverse_transform(dataY_plot)
plt.figure(figsize=(10, 6))  # plotting
plt.axvline(x=200, c='r', linestyle='--')  # size of the training set

plt.plot(dataY_plot, label='Actuall Data')  # actual plot
plt.plot(data_predict, label='Predicted Data')  # predicted plot
plt.title('Time-Series Prediction')
plt.legend()
plt.show()
