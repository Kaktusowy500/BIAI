# BIAI - Trader bot based on neural networks predictions

### Data
Stock symbols are taken from [Cboe Listed Symbols](https://www.cboe.com/us/equities/market_statistics/listed_symbols/) in CSV format.

### Nerual network
LSTM model implemeted in Pytorch

### Bot strategies
Tresh Strategy - taking sum of predictions in next few day and checking if it is lower or grater than defined threshhold

Moving average strategy - combined thresh strategy and deciding based on moving average indicator - described [here](https://wire.insiderfinance.io/earn-22-more-return-by-using-a-simple-20-50-moving-average-strategy-60b4bef7c64c)

