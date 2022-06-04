import numpy as np
import pandas as pd
from typing import Dict
from model_mock import ModelMock
import matplotlib.pyplot as plt
import datetime as dt
from trainer import TEST_PERIODS, TRAIN_PERIODS


class StockData:
    def __init__(self, amount, price):
        self.amount = amount
        self.average_price = price

    def add(self, amount, price):
        if amount > 0:
            total_val = self.amount * self.average_price + amount * price
            total_amount = self.amount + amount
            self.average_price = total_val / total_amount
            self.amount = total_amount

    def sub(self, amount, price):
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
        return self.amount * curr_price


STOCK_NAME = "AAAU"


class Trader:
    CHANGE_TRESH = 2

    def __init__(self, money, stock_prices: pd.DataFrame):
        self.model = ModelMock([1.3, -2.3, 2.1, 1.9, 1.1])
        self.bought_stocks: Dict[str, StockData] = {}
        self.money = money
        self.stocks_prices = stock_prices
        self.current_date = None
        self.wallet_history = pd.DataFrame(columns=["Cash", "Stocks_value", "Total_value"],  index=pd.to_datetime([]))

    def buy_stock(self, stock_name, money_to_spend):
        if(money_to_spend > self.money):
            raise Exception(f"Not enough money to buy {stock_name}")

        actual_price = self.stocks_prices.loc[self.current_date]["Close"]
        amount_of_stock = money_to_spend / actual_price
        self.money -= money_to_spend

        if stock_name in self.bought_stocks:
            self.bought_stocks[stock_name].add(amount_of_stock, actual_price)
        else:
            self.bought_stocks[stock_name] = StockData(
                amount_of_stock, actual_price)

        print(f"Bought {stock_name} with {money_to_spend}")

    def sell_stock(self, stock_name, amount):
        if stock_name in self.bought_stocks:
            actual_price = self.stocks_prices.loc[self.current_date]["Close"]
            value = self.bought_stocks[stock_name].sub(amount, actual_price)
            self.money += value
            print(f"Sold {amount} of {stock_name} of value {value}")

    def make_decision(self, stock_name, predictions):
        if np.sum(predictions) > self.CHANGE_TRESH:
            if self.money > 0:
                self.buy_stock(stock_name, self.money)

        elif np.sum(predictions) < -self.CHANGE_TRESH:
            self.sell_stock(stock_name, -1)

    def calc_and_save_balance(self):
        stocks_value = 0
        for stockname in self.bought_stocks.keys():
            actual_price = self.stocks_prices.loc[self.current_date]["Close"]  # Temporary for only one stock
            stocks_value += self.bought_stocks[stockname].get_current_value(actual_price)

        balance = {"Cash": self.money, "Stocks_value": stocks_value, "Total_value": stocks_value + self.money}
        self.wallet_history.loc[pd.to_datetime(self.current_date)] = balance

    def evaluate_strategy(self, start_date, end_date=None):
        self.current_date = start_date
        while(True):
            history = self.stocks_prices.loc[:start_date]
            if history.shape[0] < TRAIN_PERIODS:
                return None
            history_for_train = history[-20:]
            predictions = self.model(history_for_train)
            self.make_decision(STOCK_NAME, predictions)
            self.calc_and_save_balance()

            next_period = self.stocks_prices.loc[self.current_date:]

            if next_period.shape[0] < TEST_PERIODS + 1:
                break
            # take next date
            self.current_date = next_period.iloc[TEST_PERIODS].name
            print(self.current_date)

def plot_results(df):
    x = [dt.datetime.date(d) for d in df.index]
    fig = plt.figure(figsize=(10, 5))
    plt.title('Total wallet value in time')
    plt.ylabel('Value')
    plt.grid(True)
    plt.plot(x,
           df["Total_value"].to_numpy(),
           "b-")
    plt.savefig('plot_wallet', dpi=600)


if __name__ == "__main__":
    df = pd.read_csv('raw_data/AAAU.csv', index_col='Date', parse_dates=True)
    trader = Trader(10000, df)
    trader.evaluate_strategy("2021-01-29")
    print(trader.wallet_history)
    plot_results(trader.wallet_history)
