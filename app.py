import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_squared_log_error
import math
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pymongo
from pymongo import MongoClient
from gridfs import GridFS
import csv

def get_next_weekday(date, weekday):
    days_ahead = weekday - date.weekday()
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return date + timedelta(days_ahead)

def predict_five_days():
    # Previous code remains unchanged

    # Use the new model to generate predictions for the next five weekdays
    predicted_prices = []
    current_date = datetime.now()
    for _ in range(5):
        next_weekday = get_next_weekday(current_date, 0)  # 0 represents Monday
        if next_weekday.weekday() < 5:  # Check if the next weekday is not a weekend
            current_date = next_weekday
            last_data_point = scaled_data[-look_back:]
            prediction_input = last_data_point.reshape(1, -1)
            next_day_prediction = new_model.predict(prediction_input)
            next_day_prediction = scaler.inverse_transform(next_day_prediction)[0][0]
            predicted_prices.append((current_date.strftime("%Y-%m-%d"), next_day_prediction))

            # Update the data for the next iteration
            scaled_data = np.append(scaled_data, [next_day_prediction])

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MongoDB Atlas connection string
atlas_connection_uri = "mongodb+srv://ShunYi:Shun_30020305@smeat.zxafusy.mongodb.net/"
# Database name
db_name = "SMEAT"

# Connect to MongoDB Atlas
client = MongoClient(atlas_connection_uri)
db = client[db_name]
fs = GridFS(db)

# Load the ANN.h5 file from MongoDB Atlas
h5_file = fs.find_one({'filename': 'ANN.h5'})
if h5_file is not None:
    # Save the file locally
    with open('ANN.h5', 'wb') as file:
        file.write(h5_file.read())

# Load your existing model if it exists
try:
    model = load_model('ANN.h5')
    print("Model loaded successfully.")
except:
    model = Sequential()
    model.add(Dense(100, activation='sigmoid', input_dim=60))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    print("New model created.")

# Define the scaler object
scaler = MinMaxScaler()

# Assume this is your login logic
def authenticate(username, password):
    if username == 'admin' and password == 'password':
        return True
    return False

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/prediction')
def prediction():
    table_data = []
    with open(r'C:\Users\User\Desktop\Final\StockCode.csv', mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            table_data.append(row)

    table_data.sort(key=lambda x: x[0])

    return render_template('prediction.html', active='prediction', table_data=table_data)

@app.route('/predict', methods=['POST'])
def predict():
    symbol = request.form['symbol']
    try:
        stock = yf.Ticker(symbol)
        stock_info = stock.info  # Get stock information
        stock_name = stock_info['shortName']  # Get stock name
        stock = yf.Ticker(symbol)
        data = stock.history(period='10y')
        last_known_price = data['Close'].iloc[-1]

        if len(data) == 0:
            raise ValueError("No data available for the specified stock symbol.")

        # Fit the scaler to the data and transform it
        scaled_data = scaler.fit_transform(data[['Close']].values.reshape(-1, 1))

        # Define the look_back window and create input-output pairs
        look_back = 60
        X, y = [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:i + look_back].flatten())
            y.append(scaled_data[i + look_back][0])

        X = np.array(X)
        y = np.array(y)

        if len(X) < look_back:
            raise ValueError("Not enough data for prediction. Please choose a longer period or try another stock symbol.")

        # Create a new model
        new_model = Sequential([
            Dense(units=64, activation='sigmoid', input_dim=look_back),
            Dense(units=32, activation='tanh'),
            Dense(units=32, activation='relu'),
            Dense(units=1, activation='linear')
        ])
        new_model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the new model
        new_model.fit(X, y, batch_size=32, epochs=100, validation_split=0.1)

        # Use the new model to generate predictions for tomorrow
        last_known_price = data['Close'].iloc[-1]
        last_data_point = scaled_data[-look_back:]
        prediction_input = last_data_point.reshape(1, -1)
        tomorrow_prediction = new_model.predict(prediction_input)
        tomorrow_prediction = scaler.inverse_transform(tomorrow_prediction)[0][0]

        # Calculate the predicted price and the difference from the last known price for tomorrow
        predicted_price = tomorrow_prediction
        price_diff = predicted_price - last_known_price

        # Calculate MSE and RMSE
        mse = mean_squared_error([last_known_price], [tomorrow_prediction])
        rmse = math.sqrt(mse)

        # Generate recommendation based on the price difference
        if price_diff > 0.50:
            recommendation = "Strong Buy"
        elif 0.10 <= price_diff <= 0.50:
            recommendation = "Buy"
        elif -0.10 <= price_diff <= 0.10:
            recommendation = "Neutral"
        elif -0.50 <= price_diff < -0.10:
            recommendation = "Sell"
        else:
            recommendation = "Strong Sell"

        # Get the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return render_template('result.html', symbol=symbol, stock_name=stock_name,
                               last_known_price=last_known_price, 
                               prediction=predicted_price, recommendation=recommendation,
                               mse=mse, rmse=rmse, current_time=current_time)

    except Exception as e:
        print(f"An error occurred: {e}")
        return render_template('error.html', error_message=str(e))
    

@app.route('/prediction_five_days', methods=['POST'])
def predict_five_days():
    symbol = request.form['symbol']
    try:
        stock = yf.Ticker(symbol)
        stock_info = stock.info  # Get stock information
        stock_name = stock_info['shortName']  # Get stock name
        stock = yf.Ticker(symbol)
        data = stock.history(period='10d')  # Get historical data for the last 10 days

        if len(data) < 5:
            raise ValueError("Not enough data available for prediction. Please choose another stock symbol or try again later.")

        # Fit the scaler to the data and transform it
        scaled_data = scaler.fit_transform(data[['Close']].values.reshape(-1, 1))

        # Define the look_back window and create input-output pairs
        look_back = 5
        X, y = [], []
        for i in range(len(scaled_data) - look_back):
            X.append(scaled_data[i:i + look_back].flatten())
            y.append(scaled_data[i + look_back][0])

        X = np.array(X)
        y = np.array(y)

        if len(X) < look_back:
            raise ValueError("Not enough data for prediction. Please choose a longer period or try another stock symbol.")

        # Create a new model
        new_model = Sequential([
            Dense(units=64, activation='sigmoid', input_dim=look_back),
            Dense(units=32, activation='tanh'),
            Dense(units=32, activation='relu'),
            Dense(units=1, activation='linear')
        ])
        new_model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the new model
        new_model.fit(X, y, batch_size=32, epochs=100, validation_split=0.1)

        # Use the new model to generate predictions for the next five weekdays
        predicted_prices = []
        current_date = datetime.now()
        for _ in range(5):
        # Skip weekends
            while current_date.weekday() >= 5:  # Saturday (5) and Sunday (6)
                current_date += timedelta(days=1)
            next_weekday = current_date
            current_date += timedelta(days=1)  # Move to the next day
            last_data_point = scaled_data[-look_back:]
            prediction_input = last_data_point.reshape(1, -1)
            next_day_prediction = new_model.predict(prediction_input)
            next_day_prediction = scaler.inverse_transform(next_day_prediction)[0][0]
            predicted_prices.append((next_weekday.strftime("%Y-%m-%d"), next_day_prediction))

            # Update the data for the next iteration
            scaled_data = np.append(scaled_data, [next_day_prediction])



        # Get the current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return render_template('prediction_five_days.html', stock_name=stock_name, symbol=symbol, current_time=current_time, predicted_prices=predicted_prices)

    except Exception as e:
        print(f"An error occurred: {e}")
        return render_template('error.html', error_message=str(e))



if __name__ == '__main__':
    app.run(debug=True)
