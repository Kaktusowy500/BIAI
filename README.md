# BIAI - Trader bot based on neural networks predictions

## How to use
Clone repo
```
git clone git@github.com:Kaktusowy500/BIAI.git
```

Go to repo and install dependencies:
``` 
pip install -r requirements.txt
```
Download data by running script `download_data.py`

Pre-process data by running script `process_data.py`

Train LSTM model on chosen stock:
- in `trainer.py` script, change STOCK_NAME variable to chosen stock ticker, also other parameters such as train and test periods' length or number of epochs may be changed there
- Run `trainer.py` script
- Trained model will be saved in `models` directory

Evaluate model and strategy on test period:
- in `trader.py` script, change STOCK_NAME variable to chosen stock ticker, also strategy and strategies' parameters may be change on creation of trader instance
- Run `trader.py` script

Plots created during training and evaluation are saved in `plots` directory

## Features
### Data
Stock symbols are taken from [Cboe Listed Symbols](https://www.cboe.com/us/equities/market_statistics/listed_symbols/) in CSV format.
Data is pre-processed to create column with day to day price change

### Neural network
LSTM model implemented in Pytorch. Normalized price change and volume are used as inputs, output is equal to price change

### Bot strategies
Tresh Strategy - taking sum of predictions in next few day and checking if it is lower or grater than defined threshhold

Moving average strategy - combined thresh strategy and deciding based on moving average indicator - described [here](https://wire.insiderfinance.io/earn-22-more-return-by-using-a-simple-20-50-moving-average-strategy-60b4bef7c64c)

### Plots
Following plots are generated:
- Predicted price changes
- Predictions history compared with real price
- Wallet balance 
- Example price prediction, at the end of the training



