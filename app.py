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
import joblib
import joblib
from tensorflow.keras.models import load_model
import json
import datetime
import os

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

        # Load and preprocess data based on rice type
        if type.lower() == "regular":
            # Load the pre-trained scaler and model
            scaler = joblib.load('static/models/regular_scaler.pkl')
            # model = load_model('lstm_premium_rice_price_model.h5')
            model = load_model('static/models/lstm_regular_rice_price_model.h5', compile=False)

            df = pd.read_csv('static/datasets/reduced_regular_milled_rice.csv')
        elif type.lower() == "premium":
            # Load the pre-trained scaler and model
            scaler = joblib.load('static/models/premium_scaler.pkl')
            # model = load_model('lstm_premium_rice_price_model.h5')
            model = load_model('static/models/lstm_premium_rice_price_model.h5', compile=False)

            df = pd.read_csv('static/datasets/reduced_premium_rice.csv')
        elif type.lower() == "special":
            # Load the pre-trained scaler and model
            scaler = joblib.load('static/models/special_scaler.pkl')
            # model = load_model('lstm_premium_rice_price_model.h5')
            model = load_model('static/models/lstm_special_rice_price_model.h5', compile=False)

            df = pd.read_csv('static/datasets/reduced_special_rice.csv')
        elif type.lower() == "well milled":
            # Load the pre-trained scaler and model
            scaler = joblib.load('static/models/well_milled_scaler.pkl')
            # model = load_model('lstm_premium_rice_price_model.h5')
            model = load_model('static/models/lstm_well_milled_rice_price_model.h5', compile=False)

            df = pd.read_csv('static/datasets/reduced_well_milled_rice.csv')

        # Ensure MONTH column is integer and create DATE column
        df['MONTH'] = df['MONTH'].astype(int)
        df["DATE"] = pd.to_datetime(df['YEAR'].astype(str) + '/' + df['MONTH'].astype(str) + '/01')
        df = df.set_index('DATE').asfreq('MS')

        # Select column for price prediction
        df = df[['Price / kg']]

        progress = 30


        # Perform seasonal decomposition
        # (If this is needed for analysis but not directly part of the LSTM input, it can stay as is)
        # results = seasonal_decompose(df)  # Uncomment if you want to see seasonal decomposition

        # Split data into train and test sets
        train = df.iloc[:-12]
        test = df.iloc[-12:]



        # Scale data using the loaded scaler
        scaled_train = scaler.transform(train)

        progress = 60

        # Define generator for LSTM
        n_input = 345
        n_features = 1
        generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1)

        # Prepare for predictions
        last_train_batch = scaled_train[-n_input:]
        current_batch = last_train_batch.reshape((1, n_input, n_features))

        # Generate predictions for test set
        test_predictions = []
        for i in range(len(test)):
            current_pred = model.predict(current_batch)[0]
            test_predictions.append(current_pred)
            current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

        # Inverse transform predictions to original scale
        true_predictions = scaler.inverse_transform(test_predictions).flatten()
        test['Predictions'] = true_predictions

        progress = 70

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

        progress = 80

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


def predict_all_rice_types(num_years):
    current_year = datetime.datetime.now().year
    json_path = 'static/predictions/rice_price_predictions.json'
    
    # Check if JSON exists and if predictions are up-to-date
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            existing_data = json.load(f)
            # Get the latest year in the existing predictions
            latest_year_in_data = min(
                min(item['year'] for item in rice_data) 
                for rice_data in existing_data.values()
            )
            
            # If latest year in data is current, no need to update predictions
            if latest_year_in_data >= current_year:
                print("Predictions are up-to-date.")
                return existing_data
    
    # Prediction models and paths setup
    rice_types = {
        "regular": {
            "scaler_path": 'static/models/regular_scaler.pkl',
            "model_path": 'static/models/lstm_regular_rice_price_model.h5',
            "data_path": 'static/datasets/reduced_regular_milled_rice.csv'
        },
        "premium": {
            "scaler_path": 'static/models/premium_scaler.pkl',
            "model_path": 'static/models/lstm_premium_rice_price_model.h5',
            "data_path": 'static/datasets/reduced_premium_rice.csv'
        },
        "special": {
            "scaler_path": 'static/models/special_scaler.pkl',
            "model_path": 'static/models/lstm_special_rice_price_model.h5',
            "data_path": 'static/datasets/reduced_special_rice.csv'
        },
        "well milled": {
            "scaler_path": 'static/models/well_milled_scaler.pkl',
            "model_path": 'static/models/lstm_well_milled_rice_price_model.h5',
            "data_path": 'static/datasets/reduced_well_milled_rice.csv'
        }
    }
    
    all_predictions = {}

    for rice_type, paths in rice_types.items():
        # Load scaler, model, and data
        scaler = joblib.load(paths['scaler_path'])
        model = load_model(paths['model_path'], compile=False)
        df = pd.read_csv(paths['data_path'])
        
        # Prepare and preprocess data
        df['MONTH'] = df['MONTH'].astype(int)
        df["DATE"] = pd.to_datetime(df['YEAR'].astype(str) + '/' + df['MONTH'].astype(str) + '/01')
        df = df.set_index('DATE').asfreq('MS')
        df = df[['Price / kg']]
        
        # Scale data
        scaled_train = scaler.transform(df.iloc[:-12])
        
        # Prepare for predictions
        n_input = 345
        n_features = 1
        last_train_batch = scaled_train[-n_input:]
        current_batch = last_train_batch.reshape((1, n_input, n_features))

        # Predict for multiple years
        rice_predictions = []
        for year in range(current_year, current_year + num_years):
            yearly_predictions = []
            for month in range(1, 13):
                # Predict next month's price
                current_pred = model.predict(current_batch)[0]
                yearly_predictions.append(float(current_pred))
                
                # Update batch with the latest prediction
                current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)
            
            # Inverse transform predictions to get actual prices
            yearly_predictions = np.array(yearly_predictions).reshape(-1, 1)
            actual_yearly_predictions = scaler.inverse_transform(yearly_predictions).flatten()
            
            # Append predictions to the output list
            predicted_prices = [
                {
                    'month': calendar.month_name[month],
                    'year': year,
                    'price': round(float(price), 2)
                }
                for month, price in enumerate(actual_yearly_predictions, start=1)
            ]
            rice_predictions.extend(predicted_prices)

        # Store predictions for the current rice type
        all_predictions[rice_type] = rice_predictions
    
    # Export to JSON file
    with open(json_path, 'w') as f:
        json.dump(all_predictions, f, indent=4)

    return all_predictions

# Path to the JSON file
JSON_FILE_PATH = os.path.join('static', 'predictions', 'rice_price_predictions.json')

@app.route('/api/prices')
def get_prices():
    # Load the JSON data
    with open(JSON_FILE_PATH, 'r') as file:
        data = json.load(file)
    return jsonify(data)

# using function on load
predict_all_rice_types(num_years=5)


if __name__ == '__main__':
    app.run(debug=True)
