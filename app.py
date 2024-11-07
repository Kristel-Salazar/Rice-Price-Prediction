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
from datetime import datetime
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


# Pricing route with trigger for predict_all_rice_types if data is missing
@app.route('/pricing', methods=['POST', 'GET'])
def pricing():
    global progress
    if request.method == 'POST':
        # Initialize progress
        with progress_lock:
            progress = 10  # Start progress at 10%

        # Retrieve request parameters
        month = int(request.form.get('month'))
        month_name = calendar.month_name[month]
        rice_type = request.form.get('type').lower()
        year = int(request.form.get('year'))

        # Update progress after reading input data
        with progress_lock:
            progress = 20

        # Path to JSON file
        json_file_path = 'static/predictions/rice_price_predictions.json'

        # Load data from JSON file
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            with progress_lock:
                progress = 50  # Progress after loading JSON
        except FileNotFoundError:
            with progress_lock:
                progress = 100  # Complete progress on error
            return jsonify(error="Prediction file not found."), 404
        except json.JSONDecodeError:
            with progress_lock:
                progress = 100  # Complete progress on error
            return jsonify(error="Error decoding the JSON file."), 500

        # Get predictions for the specified rice type
        predictions = data.get(rice_type, [])
        with progress_lock:
            progress = 70  # Progress after fetching type-specific data

        # Find the matching month and year in the loaded data
        predicted_price = None
        for entry in predictions:
            if entry['month'] == month_name and entry['year'] == year:
                predicted_price = entry['price']
                break

        # If no match is found, call predict_all_rice_types to update the JSON
        if predicted_price is None:
            with progress_lock:
                progress = 80  # Indicate progress before updating JSON
            
            # Trigger the prediction update for missing data
            data = predict_all_rice_types(year)

            # Retry to fetch the predicted price after JSON update
            predictions = data.get(rice_type, [])
            for entry in predictions:
                if entry['month'] == month_name and entry['year'] == year:
                    predicted_price = entry['price']
                    break

            # If still no match is found, return an error
            if predicted_price is None:
                with progress_lock:
                    progress = 100  # Complete progress on error
                return jsonify(error=f"No prediction data available for {month_name} {year}."), 404

        # Update progress before completing
        with progress_lock:
            progress = 90  # Almost complete

        # Final progress completion
        with progress_lock:
            progress = 100  # Fully complete

        print(f'Prediction: {month_name} - {rice_type} - {predicted_price}')

        # Return the predicted price as JSON
        return jsonify(month=month_name, type=rice_type, year=year, price=predicted_price)

    # Render the template for GET requests
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


def predict_all_rice_types(end_year):
    current_year = datetime.now().year
    json_path = 'static/predictions/rice_price_predictions.json'
    
    # Check if JSON file exists and load it
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            existing_data = json.load(f)
            
        # Find the maximum year in the existing JSON file
        max_year_in_json = max(
            item['year'] for rice_type in existing_data.values() for item in rice_type
        )
        
        # Only proceed if end_year is greater than max_year_in_json
        if end_year <= max_year_in_json:
            print(f"No update needed; JSON already contains data up to {max_year_in_json}.")
            return existing_data  # Return existing data if it's already up-to-date
    
    num_years = end_year - current_year + 1
    
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

        # Predict for the range from current year to end year
        rice_predictions = []
        for year in range(current_year, end_year + 1):
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

    print("Predictions have been saved to", json_path)
    return all_predictions

# Path to the JSON file
JSON_FILE_PATH = os.path.join('static', 'predictions', 'rice_price_predictions.json')

@app.route('/api/prices')
def get_prices():
    # Load the JSON data
    with open(JSON_FILE_PATH, 'r') as file:
        data = json.load(file)
    return jsonify(data)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    end_year = data.get('end_year', datetime.now().year)

    # Call the predict_all_rice_types function with end_year as an argument
    predictions = predict_all_rice_types(end_year)

    return jsonify(predictions), 200


if __name__ == '__main__':
    app.run(debug=True)
