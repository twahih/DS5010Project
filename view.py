import ANN
import alpaca_data as api

data_api = api.alpaca_data(API_KEY = 'PKL6FLMQP9AR37P9DG3M',
                 SECRET_KEY = 'P2eGMaIyezfGgSoPjD2pbafdi0wnMwncFJfdjvsy',
                 BASE_URL = 'https://paper-api.alpaca.markets', 
                 start_date = '2022-01-01', 
                 end_date = '2022-12-31',
                 symbol = 'AAPL')


X_train, X_test, y_train, y_test = data_api.datasplit()

print(X_train)