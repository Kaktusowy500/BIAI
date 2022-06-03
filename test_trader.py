import trader
import pandas as pd
import pytest


def test_make_decision_buy_with_all_money(mocker):
    tr = trader.Trader(10000, pd.DataFrame())
    mock = mocker.patch('trader.Trader.buy_stock')
    tr.make_decision("AAAU", [1.3, -2.3, 2.1, 1.9, 1.1])
    mock.assert_called_once_with("AAAU", 10000)

