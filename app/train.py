import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import logging
from typing import Any
from app import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_processed_data(path: str) -> pd.DataFrame:
    """
    Load the processed data from a CSV file.

    Parameters:
    - path: str : Path to the processed data file.

    Returns:
    - pd.DataFrame : Loaded DataFrame.
    """
    logging.info(f"Loading processed data from {path}")
    try:
        df = pd.read_csv(path)
        logging.info(f"Successfully loaded data with shape {df.shape}")
        return df
    except FileNotFoundError as e:
        logging.error(f"File not found: {path}")
        raise e
    except Exception as e:
        logging.error(f"Error loading data from {path}: {e}")
        raise e

def train_model(train_X: pd.DataFrame, train_y: pd.Series, n_estimators: int = 100, 
                max_features: float = 0.33, max_depth: int = None, random_state: int = 42) -> RandomForestRegressor:
    """
    Train a RandomForestRegressor model.

    Parameters:
    - train_X: pd.DataFrame : Features for training.
    - train_y: pd.Series : Target variable for training.
    - n_estimators: int : The number of trees in the forest (default: 100).
    - max_features: float : The number of features to consider when looking for the best split (default: 0.33).
    - max_depth: int : The maximum depth of the tree (default: None).
    - random_state: int : Random seed (default: 42).

    Returns:
    - RandomForestRegressor : Trained model.
    """
    logging.info(f"Training RandomForestRegressor model with n_estimators={n_estimators}, "
                 f"max_features={max_features}, max_depth={max_depth}, random_state={random_state}")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(train_X, train_y)
    logging.info("Model training completed")
    return model

def save_model(model: Any, path: str):
    """
    Save the trained model to a file.

    Parameters:
    - model: Any : Trained model object to be saved.
    - path: str : Path to save the model file.
    """
    logging.info(f"Saving model to {path}")
    try:
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logging.info(f"Model saved successfully to {path}")
    except Exception as e:
        logging.error(f"Error saving model to {path}: {e}")
        raise e

def main():
    """
    Main function to load data, train the model, save the trained model, and log the experiment with MLflow.
    """
    mlflow.set_experiment("sales_forecasting")

    with mlflow.start_run():
        # Load processed training data
        train_df = load_processed_data(config.PROCESSED_TRAIN_PATH)
        
        # Separate features and target variable
        train_y = train_df['UnitSales']
        train_X = train_df.drop(columns=['UnitSales', 'DateKey'])
        
        # Define model parameters
        n_estimators = 100
        max_features = round(len(train_X.columns) / 3)
        max_depth = len(train_X.columns)
        random_state = 42

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_features", max_features)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)
        
        # Train the model
        model = train_model(train_X, train_y, n_estimators=n_estimators, max_features=max_features, 
                            max_depth=max_depth, random_state=random_state)
        
        # Log the model
        mlflow.sklearn.log_model(model, "model")
        
        # Save the model to a file
        save_model(model, config.MODEL_PATH)
        
        logging.info(f"Model trained and saved to {config.MODEL_PATH}")

if __name__ == '__main__':
    main()