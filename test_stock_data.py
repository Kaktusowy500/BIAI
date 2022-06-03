from trader import StockData
import pytest


def test_add_amount_to_zeros_data():
    stock_data = StockData(0, 0)
    AMOUNT = 100
    PRICE = 20.5

    stock_data.add(amount=AMOUNT, price=PRICE)

    assert stock_data.amount == AMOUNT
    assert stock_data.average_price == PRICE


def test_add_amount_to_already_populated_data():
    PREV_AMOUNT = 10
    PREV_AVG_PRICE = 30
    stock_data = StockData(PREV_AMOUNT, PREV_AVG_PRICE)
    AMOUNT = 20
    PRICE = 5

    stock_data.add(amount=AMOUNT, price=PRICE)

    total_amount = AMOUNT+PREV_AMOUNT
    assert stock_data.amount == total_amount
    assert stock_data.average_price == pytest.approx((
        PRICE*AMOUNT+PREV_AVG_PRICE*PREV_AMOUNT)/total_amount)


def test_sub_total_amount():
    PREV_AMOUNT = 10
    PREV_AVG_PRICE = 30
    stock_data = StockData(PREV_AMOUNT, PREV_AVG_PRICE)

    value = stock_data.sub(-1)

    assert stock_data.amount == 0
    assert stock_data.average_price == PREV_AVG_PRICE
    assert value == PREV_AMOUNT * PREV_AVG_PRICE


def test_sub_part_of_total_amount():
    PREV_AMOUNT = 10
    PREV_AVG_PRICE = 30
    stock_data = StockData(PREV_AMOUNT, PREV_AVG_PRICE)
    AMOUNT_TO_SUB = 5.5

    value = stock_data.sub(AMOUNT_TO_SUB)

    assert stock_data.amount == PREV_AMOUNT - AMOUNT_TO_SUB
    assert stock_data.average_price == PREV_AVG_PRICE
    assert value == AMOUNT_TO_SUB * PREV_AVG_PRICE


def test_sub_greater_than_total_amount():
    PREV_AMOUNT = 10
    PREV_AVG_PRICE = 30
    stock_data = StockData(PREV_AMOUNT, PREV_AVG_PRICE)
    AMOUNT_TO_SUB = 15
    
    with pytest.raises(ValueError):
        value = stock_data.sub(AMOUNT_TO_SUB)
