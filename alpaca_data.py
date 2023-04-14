
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from config import *
import os
import time

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

# Retrieve the historical data for the symbol and timeframe
bars = api.get_bars(symbol, timeframe, start=start_date, end=end_date, adjustment='raw')
