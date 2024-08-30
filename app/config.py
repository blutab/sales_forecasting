import os
import logging


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODEL_PATH = os.path.join(BASE_DIR, "models", "forecasting_model.pkl")
    DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.csv")
    PROCESSED_TRAIN_PATH = os.path.join(BASE_DIR, "data", "train_data.csv")
    PROCESSED_TEST_PATH = os.path.join(BASE_DIR, "data", "test_data.csv")
    LAGS = [7, 14, 21]
    LOGGING_LEVEL = logging.INFO
    DEBUG = True
    MODEL_NAME = "RandomForestRegressor"  # or whatever name you used in the registry
    MODEL_STAGE = "Staging"  # o
