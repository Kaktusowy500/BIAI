
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import datetime as dt
import matplotlib.pyplot as plt
from data_prep import DataPrep
from lstm import LSTM

NUM_EPOCHS = 400
LEARNING_RATE = 0.002
HIDDEN_SIZE = 50  # number of features in hidden state

TRAIN_PERIODS = 20  # number of days used for training and infering
TEST_PERIODS = 5  # number of days used for test and predict

DF_PATH = "data/YPS.csv"


def plot_results(df, predictions):
    x = [dt.datetime.date(d) for d in df.index]
    fig = plt.figure(figsize=(10, 5))
    plt.title('Price change by day')
    plt.ylabel('Price change %')
    plt.grid(True)
    plt.plot(x[:-len(predictions)],
           df[:-len(predictions)].to_numpy(),
           "b-")
    plt.plot(x[-len(predictions):],
            df[-len(predictions):].to_numpy(),
            "b--",
            label='True Values')
    plt.plot(x[-len(predictions):],
            predictions,
            "r-",
            label='Predicted Values')
    plt.legend()
    plt.savefig('plot ', dpi=600)


class Trainer():
    def __init__(self):
        self.model = LSTM(input_size=1, hidden_size=HIDDEN_SIZE, output_size=TEST_PERIODS)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)

    def train(self, x_train, y_train):
        self.model.train()
        for epoch in range(NUM_EPOCHS):
            for x, y in zip(x_train, y_train):
                y_hat, _ = self.model(x)
                self.optimizer.zero_grad()
                loss = self.criterion(y_hat, y)
                loss.backward()
                self.optimizer.step()

            if epoch % 20 == 0:
                print(f'epoch: {epoch:4} loss:{loss.item():10.8f}')

    def predict(self, data_period):
        self.model.eval()
        with torch.no_grad():
            predictions, _ = self.model(data_period)
        return predictions


if __name__ == "__main__":

    data_prep = DataPrep()
    data_prep.read_and_parse_df(DF_PATH)
    train, test = data_prep.train_test(TEST_PERIODS)
    train_scaled = data_prep.scale_data(train)
    x_train, y_train = data_prep.get_x_y_pairs(train_scaled, TRAIN_PERIODS, TEST_PERIODS)

    trainer= Trainer()
    trainer.train(x_train, y_train)
    inference_period = train_scaled[-TRAIN_PERIODS:]
    predictions = trainer.predict(inference_period)

    # Apply inverse transform to undo scaling
    predictions = data_prep.scaler.inverse_transform(np.array(predictions.reshape(-1, 1)))
    plot_results(data_prep.df, predictions)