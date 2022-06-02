from cv2 import add
import numpy as np
import pandas as pd
from typing import List, Dict
from model_mock import ModelMock


class StockData:
    def __init__(self, amount, price):
        self.amount = amount
        self.average_price = price

    def add (self, amount, price):
        total_val = self.amount * self.average_price + amount * price
        total_amount = self.amount + amount
        self.average_price = total_val / total_amount
        self.amount = total_amount 



class Trader:
    CHANGE_TRESH = 2

    def __init__(self, money, stock_prices: pd.DataFrame):
        self.model = ModelMock([1.3, -2.3, 2.1, 1.9, 1.1])
        self.bought_stocks: Dict[str, StockData] = {}
        self.money = money
        self.stocks_prices = stock_prices

    def update_stock_prices(self, stock_prices):
        self.stocks_prices = stock_prices

    def buy_stock(self, stock_name, money_to_spend):
        if(money_to_spend > self.money):
            raise Exception(f"Not enough money to buy {stock_name}")

        actual_price = self.stocks_prices["Close"].iloc[-1]    
        amount_of_stock = money_to_spend / actual_price
        self.money -= money_to_spend

        if self.bought_stocks[stock_name] == None:
            self.bought_stocks[stock_name] = StockData(self, amount_of_stock, actual_price)
        else:
            self.bought_stocks[stock_name].add(amount_of_stock , actual_price)

        print(f"Bought {stock_name} with {money_to_spend}")

    def make_decision(self, stock_name, predictions):
        if np.sum(predictions) > self.CHANGE_TRESH:
            self.buy_stock(stock_name, self.money) 
        
