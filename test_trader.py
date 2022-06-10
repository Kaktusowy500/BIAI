import trader
import pandas as pd
from strategies import TreshStrategy

# Test for a make decision and integration with TreshStrategy
def test_make_decision_buy_with_entire_money(mocker):
    tr = trader.Trader(10000, pd.DataFrame(), TreshStrategy(2))
    mock = mocker.patch('trader.Trader.buy_stock')
    STOCK_NAME = "AAAU"

    tr.make_decision(STOCK_NAME, [1.3, -2.3, 2.1, 1.9, 1.1])

    mock.assert_called_once_with(STOCK_NAME, 10000)


def test_make_decision_sell_stock_entire_amount(mocker):
    tr = trader.Trader(10000, pd.DataFrame(),TreshStrategy(2))
    mock = mocker.patch('trader.Trader.sell_stock')
    STOCK_NAME = "AAAU"

    tr.make_decision(STOCK_NAME, [-1.3, -2.3, -2.1, -1.9, 1.1])

    mock.assert_called_once_with(STOCK_NAME, -1)

def test_make_decision_hold(mocker):
    tr = trader.Trader(10000, pd.DataFrame(), TreshStrategy(2))
    mock_sell = mocker.patch('trader.Trader.sell_stock')
    mock_buy = mocker.patch('trader.Trader.buy_stock')
    STOCK_NAME = "AAAU"

    tr.make_decision(STOCK_NAME, [0, 2, -2, 0, 0.1])

    mock_sell.assert_not_called()
    mock_buy.assert_not_called()

