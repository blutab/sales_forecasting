import pandas as pd
import logging
import mlflow
from typing import Any, Tuple
from app.config import Config
from sklearn.metrics import root_mean_squared_error, mean_absolute_error

logging.basicConfig(
    level=Config.LOGGING_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s"
)

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

def evaluate_model(
    model: Any, test_X: pd.DataFrame, test_y: pd.Series
) -> Tuple[float, float]:
    logging.info("Evaluating model on test data")
    predictions = model.predict(test_X)
    rmse = root_mean_squared_error(test_y, predictions)
    mae = mean_absolute_error(test_y, predictions)
    logging.info(f"Model evaluation completed: RMSE={rmse}, MAE={mae}")

    return predictions, rmse, mae

def load_model_from_registry(model_name: str, stage: str = "Production") -> Any:
    """
    Load a model from the MLflow Model Registry.

    Parameters:
    - model_name: str : Name of the model in the registry.
    - stage: str : Stage of the model to load (default: "Production").

    Returns:
    - Any : Loaded model object.
    """
    logging.info(f"Loading {stage} model '{model_name}' from MLflow Model Registry")
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{stage}")
        logging.info(f"Successfully loaded {stage} model '{model_name}' from MLflow Model Registry")
        return model
    except mlflow.exceptions.MlflowException as e:
        logging.error(f"Error loading model '{model_name}' from MLflow Model Registry: {e}")
        raise e

def save_model(model: Any, path: str):
    """
    Save the trained model to a file.

    Parameters:
    - model: Any : Trained model object to be saved.
    - path: str : Path to save the model file.
    """
    logging.info(f"Saving model to {path}")
    try:
        mlflow.sklearn.save_model(model, path)
        logging.info(f"Model saved successfully to {path}")
    except Exception as e:
        logging.error(f"Error saving model to {path}: {e}")
        raise e