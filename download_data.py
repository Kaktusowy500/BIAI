import yfinance as yf
import pandas as pd
import os

try:
    os.mkdir('raw_data')

    symbols_df = pd.read_csv('bats_symbols.csv')
    tickers = symbols_df['Name'].to_list()
    tickers_str = ' '.join(tickers)

    for ticker in tickers:
        yf_df = yf.download(ticker, start='2019-01-01', end="2021-12-30")
        yf_df.to_csv('raw_data/' + ticker + '.csv')

except FileExistsError:
    pass