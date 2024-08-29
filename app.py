from flask import Flask, request, jsonify
import pandas as pd
import logging
from app.inference import load_model, convert_log_to_units
from app.data_preparation import add_lagged_features
from app import config

# Set up Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the model once when the app starts
model = load_model(config.MODEL_PATH)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input data from request
        input_data = request.get_json()

        # Convert the input data to a DataFrame
        input_df = pd.DataFrame([input_data])

        # Ensure the input data has the required columns
        required_columns = ['StoreCount', 'ShelfCapacity', 'PromoShelfCapacity', 'IsPromo',
                            'ItemNumber', 'CategoryCode', 'GroupCode', 'month', 'weekday', 'UnitSales', 'DateKey']

        missing_columns = [col for col in required_columns if col not in input_df.columns]
        if missing_columns:
            return jsonify({"error": f"Missing columns in input data: {missing_columns}"}), 400

        # Preprocessing steps: Convert DateKey to datetime and generate month and weekday features
        input_df['DateKey'] = pd.to_datetime(input_df['DateKey'], format='%Y%m%d')
        input_df['month'] = input_df['DateKey'].dt.month.astype('category')
        input_df['weekday'] = input_df['DateKey'].dt.weekday.astype('category')
        input_df['GroupCode'] = input_df['GroupCode'].astype('category')
        input_df['ItemNumber'] = input_df['ItemNumber'].astype('category')
        input_df['CategoryCode'] = input_df['CategoryCode'].astype('category')

        # Add lagged features
        lags = [7, 14, 21]
        input_df = add_lagged_features(input_df, lags, 'UnitSales')

        # Check if lagged features were added successfully
        lagged_columns = [f'UnitSales_lag_{lag}' for lag in lags]
        missing_lagged_columns = [col for col in lagged_columns if col not in input_df.columns]
        if missing_lagged_columns:
            return jsonify({"error": f"Missing lagged columns after processing: {missing_lagged_columns}"}), 500

        # Ensure that the dataframe has all the necessary columns
        if input_df.empty:
            return jsonify({"error": "Input data resulted in empty DataFrame after processing."}), 400

        # Make a prediction
        prediction = model.predict(input_df)[0]

        # Convert prediction from log scale to units
        prediction_units = convert_log_to_units(prediction)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction_units})

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)