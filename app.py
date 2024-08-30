from flask import Flask, request, jsonify
import pandas as pd
import logging
from app.config import Config
from app.inference import Inferencer
from app.data_preparation import DataPreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class App:
    def __init__(self, config: Config):
        self.app = Flask(__name__)
        self.config = config
        self.inferencer = Inferencer(config)
        self.preprocessor = DataPreprocessor(config)

        self.setup_routes()
    
    def setup_routes(self):
        self.app.route('/predict', methods=['POST'])(self.predict)
    
    def predict(self):
        try:
            input_data = request.get_json()
            input_df = pd.DataFrame([input_data])
            self.validate_input_data(input_df)
            
            processed_df = self.preprocessor.preprocess_data(input_df)
            processed_df = self.preprocessor.add_lagged_features(processed_df, 'UnitSales')

            # Check if the processed_df is empty after preprocessing
            if processed_df.empty:
                return jsonify({"error": "Not enough data after preprocessing to make a prediction."}), 400
            
            
            prediction_units = self.inferencer.get_predictions(processed_df)
            
            return jsonify({'prediction': prediction_units})
        
        except Exception as e:
            logging.error(f"Error during prediction: {e}")
            return jsonify({"error": str(e)}), 500
    
    def validate_input_data(self,input_df: pd.DataFrame) -> None:
        required_columns = [
            'StoreCount', 'ShelfCapacity', 'PromoShelfCapacity', 'IsPromo',
            'ItemNumber', 'CategoryCode', 'GroupCode', 'UnitSales', 'DateKey'
        ]
        missing_columns = [col for col in required_columns if col not in input_df.columns]
        if missing_columns:
            raise KeyError(f"Missing required columns: {', '.join(missing_columns)}")

    
def create_app(config: Config) -> Flask:
    app = App(config)
    return app.app

if __name__ == '__main__':
    app = create_app(Config)
    app.run(host="0.0.0.0", port=5000, debug=True)