from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import calendar 
import threading
import time 

app = Flask(__name__)

# Global variable to store progress
progress = 0

# Define a lock to ensure thread-safe access to the progress variable
progress_lock = threading.Lock()

# Define a route for the web interface
@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index.html')


# Pricing route
@app.route('/pricing', methods=['POST', 'GET'])
def pricing():
    global progress
    if request.method == 'POST':

        month = int(request.form.get('month'))
        month_name = calendar.month_name[month]
        type = request.form.get('type')
        year = int(request.form.get('year'))

        print(request.form)

        with progress_lock:
            progress = 10  # Start progress at 10%

        # Load and preprocess data
        # loading data by rice type
        if type.lower() == "regular":
            df = pd.read_csv('static/datasets/reduced_regular_milled_rice.csv')
        elif type.lower() == "premium":
            df = pd.read_csv('static/datasets/reduced_premium_rice.csv')
        elif type.lower() == "special":
            df = pd.read_csv('static/datasets/reduced_special_rice.csv')
        elif type.lower() == "well milled":
            df = pd.read_csv('datasets/redunced_well_milled.csv')

        df['MONTH'] = df['MONTH'].astype(int)
        df["DATE"] = pd.to_datetime(df['YEAR'].astype(str) + '/' + df['MONTH'].astype(str) + '/01')
        df = df.set_index('DATE').asfreq('MS')

        with progress_lock:
            progress = 20  # Update progress after data load

        # Select column for price prediction
        df = df[['Price / kg']]

        # Perform seasonal decomposition
        results = seasonal_decompose(df)

        # Split data into train and test sets
        train = df.iloc[:-12]
        test = df.iloc[-12:]

        with progress_lock:
            progress = 30  # Update progress after preprocessing

        # Scale data
        scaler = MinMaxScaler()
        scaler.fit(train)
        scaled_train = scaler.transform(train)

        with progress_lock:
            progress = 40  # Update progress after decomposition


        # Define generator for LSTM
        n_input = 345
        n_features = 1
        generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

        with progress_lock:
            progress = 50  # Update after scaling

        model = Sequential([
            LSTM(100, activation='relu', input_shape=(n_input, n_features)),
            Dense(1)
        ])
        with progress_lock:
            progress = 60  # Update after model definition

        model.compile(optimizer='adam', loss='mse')
        # Fit the model
        model.fit(generator, epochs=50)

        with progress_lock:
            progress = 80  # Update after model training

        # Prepare for predictions
        last_train_batch = scaled_train[-n_input:]
        current_batch = last_train_batch.reshape((1, n_input, n_features))

        with progress_lock:
            progress = 90  # Update after generating test predictions

        # Generate predictions for test set
        test_predictions = []
        for i in range(len(test)):
            current_pred = model.predict(current_batch)[0]
            test_predictions.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        # Inverse transform predictions to original scale
        true_predictions = scaler.inverse_transform(test_predictions).flatten()
        test['Predictions'] = true_predictions

        # Calculate evaluation metrics
        rmse = sqrt(mean_squared_error(test['Price / kg'], test['Predictions']))
        mse = mean_squared_error(test['Price / kg'], test['Predictions'])
        mae = mean_absolute_error(test['Price / kg'], test['Predictions'])
        print(f'RMSE: {rmse}, MSE: {mse}, MAE: {mae}')

        # Predict future price for the selected month and year
        future_dates = pd.date_range(start=f'{year}-{month}-01', periods=1, freq='MS')
        future_predictions = []
        for _ in range(len(future_dates)):
            current_pred = model.predict(current_batch)[0]
            future_predictions.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        # Inverse transform future predictions
        actual_future_predictions = scaler.inverse_transform(future_predictions)
        predicted_price = actual_future_predictions[0][0]
        print(f"Predicted Price for {future_dates[0].strftime('%B %Y')}: {predicted_price}")

        # Final progress completion
        with progress_lock:
            progress = 100

        # Return the data as JSON
        predicted_price = round(actual_future_predictions[0][0], 2)  # Format to 2 decimal places
        print(f'{month} - {type} - {year} {predicted_price}')
        return jsonify(month=month_name, type=type, year=year, price=predicted_price)

    return render_template('Pricing.html')


# About route
@app.route('/about', methods=['POST', 'GET'])
def about():
    return render_template('About.html')


@app.route('/start-task', methods=['POST'])
def start_task():
    thread = threading.Thread(target=pricing)
    thread.start()
    return jsonify({"status": "Task started"})

@app.route('/progress')
def get_progress():
    global progress
    return jsonify(progress=progress)


if __name__ == '__main__':
    app.run(debug=True)
