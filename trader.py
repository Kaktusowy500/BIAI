from cv2 import add
from matplotlib.pyplot import hist
import numpy as np
import pandas as pd
from typing import List, Dict
from model_mock import ModelMock
from trainer import TEST_PERIODS, TRAIN_PERIODS


class StockData:
    def __init__(self, amount, price):
        self.amount = amount
        self.average_price = price

    def add(self, amount, price):
        total_val = self.amount * self.average_price + amount * price
        total_amount = self.amount + amount
        self.average_price = total_val / total_amount
        self.amount = total_amount


STOCK_NAME = "AAAU"


class Trader:
    CHANGE_TRESH = 2

    def __init__(self, money, stock_prices: pd.DataFrame):
        self.model = ModelMock([1.3, -2.3, 2.1, 1.9, 1.1])
        self.bought_stocks: Dict[str, StockData] = {}
        self.money = money
        self.stocks_prices = stock_prices
        self.current_date = None

    def buy_stock(self, stock_name, money_to_spend):
        if(money_to_spend > self.money):
            raise Exception(f"Not enough money to buy {stock_name}")

        actual_price = self.stocks_prices.loc[self.current_date]["Close"]
        amount_of_stock = money_to_spend / actual_price
        self.money -= money_to_spend

        if stock_name in self.bought_stocks:
            self.bought_stocks[stock_name].add(amount_of_stock, actual_price)
        else:
            self.bought_stocks[stock_name] = StockData(amount_of_stock, actual_price)

        print(f"Bought {stock_name} with {money_to_spend}")

    def make_decision(self, stock_name, predictions):
        if self.money > 0: 
            if np.sum(predictions) > self.CHANGE_TRESH:
                self.buy_stock(stock_name, self.money)

    def evaluate_strategy(self, start_date, end_date=None):
        self.current_date = start_date
        while(True):
            history = self.stocks_prices.loc[:start_date]
            if history.shape[0] < TRAIN_PERIODS:
                return None
            history_for_train = history[-20:]
            predictions = self.model(history_for_train)
            self.make_decision(STOCK_NAME, predictions)

            next_period = self.stocks_prices.loc[self.current_date:]

            if next_period.shape[0] < TEST_PERIODS + 1:
                break
            # take next date
            self.current_date = next_period.iloc[TEST_PERIODS].name
            print(self.current_date)


if __name__ == "__main__":
    df = pd.read_csv('raw_data/AAAU.csv', index_col='Date', parse_dates=True)
    trader = Trader(10000, df)
    trader.evaluate_strategy("2021-01-29")
