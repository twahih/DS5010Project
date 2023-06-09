import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi

from sklearn.model_selection import train_test_split

# Set up API credentials
API_KEY = 'PKL6FLMQP9AR37P9DG3M'
SECRET_KEY = 'P2eGMaIyezfGgSoPjD2pbafdi0wnMwncFJfdjvsy'
BASE_URL = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(API_KEY, SECRET_KEY, base_url=BASE_URL, api_version='v2')

symbol = 'AAPL'
timeframe = tradeapi.rest.TimeFrame.Day  # 1 day

# Define the start and end dates for the historical data
start_date = '2022-01-01'
end_date = '2022-12-31'


symbol = 'AAPL'
timeframe = tradeapi.rest.TimeFrame.Day  # 1 day

# Define the start and end dates for the historical data
start_date = '2022-01-01'
end_date = '2022-12-31'

# Retrieve the historical data for the symbol and timeframe
bars = api.get_bars(symbol, timeframe, start=start_date, end=end_date, adjustment='raw')

# Convert the response to a Pandas DataFrame
data = pd.DataFrame([{
    'timestamp': bar.t,
    'open': bar.o,
    'high': bar.h,
    'low': bar.l,
    'close': bar.c,
    'volume': bar.v
} for bar in bars])


data.reset_index()
data = data.drop('timestamp', axis=1)
data.head()
data['close_diff'] = data['close'].diff()

# Create a new column 'indicator' with 1 if the difference is positive, and 0 if it is negative or NaN
data['indicator'] = (data['close_diff'] > 0).astype(int)

# Drop rows with NaN values
data = data.dropna()

# Reset the index after dropping rows
df = data.reset_index(drop=True)

# Print the DataFrame with the NaN row removed
print(data.head())


y = data['indicator']
X=data.drop(['indicator'], axis = 1)

print(X,y)
#X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.1)


