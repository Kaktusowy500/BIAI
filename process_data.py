import pandas as pd
import numpy as np
import os

raw_data_tickers = os.listdir('raw_data')
tickers = [raw_data_ticker.split('.')[0] for raw_data_ticker in raw_data_tickers]

try:
    os.mkdir('data')

    for ticker in tickers:
        yf_df = pd.read_csv('raw_data/' + ticker + '.csv')

        if yf_df.shape[0] >= 200:
            yf_df = yf_df[["Date", "Close"]]
            yf_df.rename(columns={"Close": "Price %"}, inplace=True)

            price_np = yf_df["Price %"].to_numpy()
            price_np = np.diff(price_np) / price_np[1:]
            
            yf_df = yf_df.iloc[1:, :]
            yf_df["Price %"] = price_np

            yf_df.to_csv('data/' + ticker + '.csv')

except FileExistsError:
    pass