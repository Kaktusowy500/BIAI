import pandas as pd
import numpy as np
import os

raw_data_tickers = os.listdir('raw_data')
tickers = [raw_data_ticker.split('.')[0] for raw_data_ticker in raw_data_tickers]

try:
    os.mkdir('data')

    for ticker in tickers:
        yf_df = pd.read_csv('raw_data/' + ticker + '.csv')

        if yf_df.shape[0] >= 240 * 3:
            yf_df = yf_df[["Date", "Close", "Volume"]]
            yf_df["Price"] = yf_df["Close"]
            yf_df.rename(columns={"Close": "Price %", "Volume": "Volume %"}, inplace=True)

            prices = yf_df["Price %"].to_numpy()
            prices = np.diff(prices) / prices[1:]

            volumes = yf_df["Volume %"].to_numpy()
            volumes[volumes == 0] = 1
            volumes = np.diff(volumes) / volumes[1:]
            
            yf_df = yf_df.iloc[1:, :]
            yf_df["Price %"] = prices
            yf_df["Volume %"] = volumes

            yf_df.to_csv('data/' + ticker + '.csv')


except FileExistsError:
    print('Data directory exists, if you want to process new data just remove it!')