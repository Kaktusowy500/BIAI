import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
import torch
from collections import OrderedDict

from typing import Dict
from model_mock import ModelMock
from lstm import LSTM
from trainer import TEST_PERIODS, TRAIN_PERIODS, HIDDEN_SIZE, INPUT_SIZE, Trainer
from strategies import Strategy, TreshStrategy, Decision, MovingAvgStrategy
from data_prep import DataPrep

class StockData:
    def __init__(self, amount, price):
        self.amount = amount
        self.average_price = price

    def add(self, amount, price):
        """Adds amount of stock, bought at defined price"""
        if amount > 0:
            total_val = self.amount * self.average_price + amount * price
            total_amount = self.amount + amount
            self.average_price = total_val / total_amount
            self.amount = total_amount

    def sub(self, amount, price):
        """Substracts amount of stock, returns its value basing on current price"""
        if amount < 0:
            value = self.amount * price
            self.amount = 0
            return value

        if amount > self.amount:
            raise ValueError(f"Too low stock amount")

        value = amount * price
        self.amount -= amount
        return value

    def get_current_value(self, curr_price):
        """Returns current value of holded stock"""
        return self.amount * curr_price


STOCK_NAME = "AAAU"


class Trader:

    def __init__(self, money, stock_name, strategy: Strategy):
        self.model = LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, output_size=TEST_PERIODS)
        self.data_prep = DataPrep()
        self.bought_stocks: Dict[str, StockData] = {}
        self.money = money
        self.stock_name = stock_name
        self.stocks_prices = self.data_prep.read_and_parse_df(f'data/{stock_name}.csv', dt_from='2021-01-01')
        print(self.stocks_prices)
        self.current_date = None
        self.strategy = strategy
        self.predictions_history = []
        self.wallet_history = pd.DataFrame(columns=["Cash", "Stocks_value", "Total_value"], index=pd.to_datetime([]))

        self.model.load_state_dict(torch.load(f'models/{self.stock_name}'))

    def buy_stock(self, stock_name, money_to_spend):
        """Buys stock with defined amount of money"""
        if(money_to_spend > self.money):
            raise Exception(f"Not enough money to buy {stock_name}")

        actual_price = self.stocks_prices.loc[self.current_date]["Price"]
        amount_of_stock = money_to_spend / actual_price
        self.money -= money_to_spend

        if stock_name in self.bought_stocks:
            self.bought_stocks[stock_name].add(amount_of_stock, actual_price)
        else:
            self.bought_stocks[stock_name] = StockData(
                amount_of_stock, actual_price)

        print(f"Bought {stock_name} with {money_to_spend}")

    def sell_stock(self, stock_name, amount):
        """Sells amount of stocks"""
        if stock_name in self.bought_stocks:
            actual_price = self.stocks_prices.loc[self.current_date]["Price"]
            value = self.bought_stocks[stock_name].sub(amount, actual_price)
            self.money += value
            print(f"Sold {amount} of {stock_name} of value {value}")

    def make_decision(self, stock_name, predictions):
        """Decides about buying, selling and holding basing on predictions"""
        result = self.strategy.execute(predictions, self.stocks_prices, self.current_date)
        if result == Decision.buy:
            if self.money > 0:
                self.buy_stock(stock_name, self.money)
        elif result == Decision.sell:
            self.sell_stock(stock_name, -1)

    def calc_and_save_balance(self):
        """Calcs actual wallet balance and saves into dataframe"""
        stocks_value = 0
        for stockname in self.bought_stocks.keys():
            actual_price = self.stocks_prices.loc[self.current_date]["Price"]  # Temporary for only one stock
            stocks_value += self.bought_stocks[stockname].get_current_value(actual_price)

        balance = {"Cash": self.money, "Stocks_value": stocks_value, "Total_value": stocks_value + self.money}
        self.wallet_history.loc[pd.to_datetime(self.current_date)] = balance

    def evaluate_strategy(self, start_date, end_date=None):
        """Evaluates strategy basing on historical data"""
        self.current_date = start_date
        while(True):
            history = self.stocks_prices.loc[:self.current_date]
            if history.shape[0] < TRAIN_PERIODS:
                return None
            history_for_inference = history[-TRAIN_PERIODS:]
            history_for_inference = self.data_prep.scale_data(history_for_inference.values)
            
            self.model.eval()
            with torch.no_grad():
                predictions, _ = self.model(history_for_inference)
            
            predictions = predictions.numpy()
            predictions = predictions.reshape(-1, 1)
            predictions = self.data_prep.scaler_price.inverse_transform(predictions)
            predictions = predictions.ravel()

            # Save prediction history
            current_datetime = history.iloc[-2].name
            dates = pd.date_range(start=current_datetime, periods=6, freq='B')
            price_history = [history.iloc[-2]['Price'], history.iloc[-2]['Price'] * (1 + predictions[0])]

            for i in range(1, len(predictions)):
                print(i)
                price_history.append(price_history[i - 1] * (1 + predictions[i]))

            prediction_history = pd.DataFrame(columns=['Price'], index=pd.to_datetime(dates))
            prediction_history['Price'] = price_history
            self.predictions_history.append(prediction_history)

            print('Predictions: ', predictions)
            self.make_decision(STOCK_NAME, predictions)
            self.calc_and_save_balance()

            next_period = self.stocks_prices.loc[self.current_date:]

            if next_period.shape[0] < TEST_PERIODS + 1:
                break
            # take next date
            self.current_date = next_period.iloc[TEST_PERIODS].name
            print(self.current_date)


def plot_wallet_history(df):
    """Total wallet value in time"""
    x = [dt.datetime.date(d) for d in df.index]
    fig = plt.figure(figsize=(10, 5))
    plt.clf()
    plt.title('Total wallet value in time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.plot(x,
           df["Total_value"].to_numpy(),
           "b-")
    plt.savefig('plots/plot_wallet', dpi=600)

def plot_predictions_history(predictions_history):
    plt.clf()
    plt.title("Prediction history")
    plt.grid(True)
    plt.ylabel('Price')
    plt.xlabel('Date')

    for prediction in predictions_history:
        x = [dt.datetime.date(d) for d in prediction.index]
        plt.plot(x, prediction['Price'].to_numpy(), "b-", label='Predicted')

    plt.plot(trader.stocks_prices['Price'], "r-", label='Real')
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.savefig('plots/predictions_history', dpi=600)

if __name__ == "__main__":
    trader = Trader(10000, 'WIL', TreshStrategy(0.01))
    trader.evaluate_strategy("2021-02-10")
    print(trader.wallet_history)
    plot_wallet_history(trader.wallet_history)
    plot_predictions_history(trader.predictions_history)
