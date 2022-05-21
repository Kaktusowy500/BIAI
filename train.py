from lstm import CustomLSTM
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preparer import DataPreparer

NUM_EPOCHS = 1000
LEARNING_RATE = 0.001

INPUT_SIZE = 5  # number of features
HIDDEN_SIZE = 2  # number of features in hidden state
NUM_LAYERS = 1  # number of stacked lstm layers

NUM_CLASSES = 1  # number of output classes

def plot_results(predicted_data, ground_truth):
  data_predict = predicted_data.data.numpy()  # numpy conversion
  dataY_plot = ground_truth.data.numpy()

  data_predict = data_prep.mm_inverse_transform(data_predict)
  dataY_plot =  data_prep.mm_inverse_transform(dataY_plot)

  plt.figure(figsize=(10, 6))  # plotting
  plt.axvline(x=200, c='r', linestyle='--')  # size of the training set

  plt.plot(dataY_plot, label='Actuall Data')  # actual plot
  plt.plot(data_predict, label='Predicted Data')  # predicted plot
  plt.title('Time-Series Prediction')
  plt.legend()
  plt.show()


df = pd.read_csv("SBUX.csv", index_col='Date', parse_dates=True)

data_prep = DataPreparer()
data_prep.split_and_convert_to_tensors(df)

X_train_tensors_final = torch.reshape(data_prep.X_train_tensor, (data_prep.X_train_tensor.shape[0], 1, data_prep.X_train_tensor.shape[1]))
X_test_tensors_final = torch.reshape(data_prep.X_test_tensor, (data_prep.X_test_tensor.shape[0], 1, data_prep.X_test_tensor.shape[1]))

print("Training Shape", X_train_tensors_final.shape, data_prep.y_train_tensor.shape)
print("Testing Shape", X_test_tensors_final.shape, data_prep.y_test_tensor.shape)

lstm = CustomLSTM(NUM_CLASSES, INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, X_train_tensors_final.shape[1])
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=LEARNING_RATE)

for epoch in range(NUM_EPOCHS):
    outputs = lstm.forward(X_train_tensors_final)  # forward pass
    optimizer.zero_grad()  # caluclate the gradient, manually setting to 0

    # obtain the loss function
    loss = criterion(outputs, data_prep.y_train_tensor)

    loss.backward()  # calculates the loss of the loss function

    optimizer.step()  # improve from loss, i.e backprop
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, loss: {loss.item():.5f}")


df_X_ss, df_y_mm = data_prep.normalize_df_to_tensor(df)
# reshaping the dataset
df_X_ss = torch.reshape(df_X_ss, (df_X_ss.shape[0], 1, df_X_ss.shape[1]))

train_predict = lstm(df_X_ss)  # forward pass
plot_results(train_predict, df_y_mm)
