from strategies import MovingAvgStrategy, Decision
import pandas as pd

def test_moving_avg_strategy_buy_signal(mocker):
    strategy = MovingAvgStrategy(2, short_period = 5, long_period = 20)
    mock_ma = mocker.patch('strategies.MovingAvgStrategy.moving_avg_prediction', return_value = Decision.buy)
    mock_tr = mocker.patch('strategies.TreshStrategy.execute', return_value = Decision.buy)
    PREDICTIONS = [1.3, -2.3, 2.1, 1.9, 1.1]
    CURRENT_DATE = None

    STOCK_HISTORY = pd.DataFrame()

    decision = strategy.execute(PREDICTIONS, STOCK_HISTORY, CURRENT_DATE)

    mock_ma.assert_called_once_with(STOCK_HISTORY)
    mock_tr.assert_called_once_with(PREDICTIONS, STOCK_HISTORY, CURRENT_DATE)
    assert decision == Decision.buy


def test_moving_avg_strategy_hold_signal(mocker):
    strategy = MovingAvgStrategy(2, short_period = 5, long_period = 20)
    mock_ma = mocker.patch('strategies.MovingAvgStrategy.moving_avg_prediction', return_value = Decision.sell)
    mock_tr = mocker.patch('strategies.TreshStrategy.execute', return_value = Decision.hold)
    PREDICTIONS = [1.3, -2.3, 2.1, 1.9, 1.1]
    CURRENT_DATE = None

    STOCK_HISTORY = pd.DataFrame()

    decision = strategy.execute(PREDICTIONS, STOCK_HISTORY, CURRENT_DATE)

    mock_ma.assert_called_once_with(STOCK_HISTORY)
    mock_tr.assert_called_once_with(PREDICTIONS, STOCK_HISTORY, CURRENT_DATE)
    assert decision == Decision.hold


