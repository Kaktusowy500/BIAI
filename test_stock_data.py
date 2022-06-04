from trader import StockData
import pytest


@pytest.mark.parametrize("amount_add, curr_price", [(100, 20), (0, 0), (1.5, 0)])
def test_add_amount_to_zeros_data(amount_add, curr_price):
    stock_data = StockData(0, 0)

    stock_data.add(amount=amount_add, price=curr_price)

    assert stock_data.amount == amount_add
    assert stock_data.average_price == curr_price


@pytest.mark.parametrize("amount_add, curr_price", [(100, 20), (0, 0), (1.5, 0)])
def test_add_amount_to_already_populated_data(amount_add, curr_price):
    PREV_AMOUNT = 10
    PREV_AVG_PRICE = 30
    stock_data = StockData(PREV_AMOUNT, PREV_AVG_PRICE)

    stock_data.add(amount=amount_add, price=curr_price)

    total_amount = amount_add + PREV_AMOUNT
    assert stock_data.amount == total_amount
    assert stock_data.average_price == pytest.approx((
        curr_price * amount_add + PREV_AVG_PRICE * PREV_AMOUNT) / total_amount)


def test_sub_total_amount():
    PREV_AMOUNT = 10
    PREV_AVG_PRICE = 30
    stock_data = StockData(PREV_AMOUNT, PREV_AVG_PRICE)
    CURRENT_PRICE = 20

    value = stock_data.sub(-1, CURRENT_PRICE)

    assert stock_data.amount == 0
    assert stock_data.average_price == PREV_AVG_PRICE
    assert value == PREV_AMOUNT * CURRENT_PRICE


@pytest.mark.parametrize("amount_sub, curr_price", [(10, 30), (10, 0), (5.2, 3.2), (0, 2), (2, 0)])
def test_sub_part_of_total_amount(amount_sub, curr_price):
    PREV_AMOUNT = 10
    PREV_AVG_PRICE = 30
    stock_data = StockData(PREV_AMOUNT, PREV_AVG_PRICE)

    value = stock_data.sub(amount_sub, curr_price)

    assert stock_data.amount == PREV_AMOUNT - amount_sub
    assert stock_data.average_price == PREV_AVG_PRICE
    assert value == amount_sub * curr_price


def test_sub_greater_than_total_amount():
    PREV_AMOUNT = 10
    PREV_AVG_PRICE = 30
    stock_data = StockData(PREV_AMOUNT, PREV_AVG_PRICE)
    AMOUNT_TO_SUB = 15
    CURRENT_PRICE = 20

    with pytest.raises(ValueError):
        value = stock_data.sub(AMOUNT_TO_SUB, CURRENT_PRICE)
