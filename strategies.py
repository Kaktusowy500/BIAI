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
    def execute(self, stock_name, predictions, stock_history) -> str:
        pass

class TreshStrategy(Strategy):
    def __init__(self, change_tresh):
        super().__init__()
        self.change_tresh = change_tresh

    def execute(self, stock_name, predictions, stock_history) -> str:
        if np.sum(predictions) > self.change_tresh:
            return Decision.buy
        elif np.sum(predictions) < -self.change_tresh:
            return Decision.sell

        return Decision.hold
    
