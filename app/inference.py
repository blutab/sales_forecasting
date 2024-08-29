import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import mlflow
import logging
from typing import Tuple, Any
from app import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_processed_data(path: str) -> pd.DataFrame:
    logging.info(f"Loading processed test data from {path}")
    try:
        df = pd.read_csv(path)
        logging.info(f"Successfully loaded test data with shape {df.shape}")
        return df
    except FileNotFoundError as e:
        logging.error(f"File not found: {path}")
        raise e
    except Exception as e:
        logging.error(f"Error loading test data from {path}: {e}")
        raise e

def load_model(path: str) -> Any:
    logging.info(f"Loading model from {path}")
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded successfully")
        return model
    except FileNotFoundError as e:
        logging.error(f"Model file not found: {path}")
        raise e
    except Exception as e:
        logging.error(f"Error loading model from {path}: {e}")
        raise e

def evaluate_model(model: Any, test_X: pd.DataFrame, test_y: pd.Series) -> Tuple[float, float]:
    logging.info("Evaluating model on test data")
    predictions = model.predict(test_X)
    rmse = mean_squared_error(test_y, predictions, squared=False)
    mae = mean_absolute_error(test_y, predictions)
    logging.info(f"Model evaluation completed: RMSE={rmse}, MAE={mae}")
    
    # Log metrics with MLflow
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    
    return rmse, mae

def convert_log_to_units(prediction: float) -> int:
    return round(math.exp(prediction))

def main():
    with mlflow.start_run():
        # Load processed test data
        test_df = load_processed_data(config.PROCESSED_TEST_PATH)
        
        # Separate features and target variable
        test_y = test_df['UnitSales']
        test_X = test_df.drop(columns=['UnitSales', 'DateKey'])
        
        # Load the trained model
        model = load_model(config.MODEL_PATH)
        
        # Evaluate the model
        rmse, mae = evaluate_model(model, test_X, test_y)
        
        # Example prediction and logging results
        example_pred = model.predict(test_X.head(1))
        logging.info(f"Example prediction (log scale): {example_pred[0]}")
        logging.info(f"Predicted UnitSales: {convert_log_to_units(example_pred[0])}")

if __name__ == '__main__':
    main()