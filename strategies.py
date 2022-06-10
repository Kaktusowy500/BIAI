import numpy as np
from abc import ABC, abstractmethod
import enum

class Decision(enum.Enum):
    buy = 1
    sell = 2
    hold = 3


class Strategy(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def execute(self, predictions, stock_history, current_date) -> Decision:
        pass

class TreshStrategy(Strategy):
    def __init__(self, change_tresh):
        super().__init__()
        self.change_tresh = change_tresh

    def execute(self, predictions, stock_history, current_date) -> Decision:
        if np.sum(predictions) > self.change_tresh:
            return Decision.buy
        elif np.sum(predictions) < -self.change_tresh:
            return Decision.sell

        return Decision.hold

class MovingAvgStrategy(Strategy):
    def __init__(self, change_tresh, short_period, long_period):
        super().__init__()
        self.change_tresh = change_tresh
        self.short_period = short_period
        self.long_period = long_period
        self.current_date = None
        self.tresh_strategy = TreshStrategy(change_tresh)

    def execute(self, predictions, stock_history, current_date) -> Decision:
        self.current_date = current_date
        ma_decision = self.moving_avg_prediction(stock_history)
        tresh_decision = self.tresh_strategy.execute(predictions, stock_history, current_date)
        if tresh_decision == ma_decision:
            return ma_decision
        return Decision.hold



    def moving_avg_prediction(self, df):
        df = df.loc[:self.current_date].copy()
        df['long_MA'] = df['Close'].rolling(int(self.long_period)).mean()
        df['short_MA'] = df['Close'].rolling(int(self.short_period)).mean()
        df['crosszero'] = np.where(df['short_MA'] > df['long_MA'], 1.0, 0.0)
        df['direction'] = df['crosszero'].diff()
        if df['direction'].iloc[-self.short_period:].isin([1.0]).any().any():
            return Decision.buy
        elif df['direction'].iloc[-self.short_period:].isin([-1.0]).any().any():
            return Decision.sell
        return Decision.hold


        