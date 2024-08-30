import pandas as pd
import numpy as np
import mlflow
import logging
from app.config import Config
from app.utils import load_model, evaluate_model, load_processed_data

# Set up logging
logging.basicConfig(
    level=Config.LOGGING_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Inferencer:
    def __init__(self, config: Config):
        self.config = config
        self.model = load_model(Config.MODEL_PATH)

    def convert_log_to_units(self, prediction: float) -> int:
        return np.round(np.exp(prediction))

    def get_predictions(self, test_df: pd.DataFrame):
        with mlflow.start_run():
            # Separate features and target variable
            test_y = test_df["UnitSales"]
            test_X = test_df.drop(columns=["UnitSales", "DateKey"])

            # Evaluate the model
            predictions, rmse, mae = evaluate_model(self.model, test_X, test_y)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

            converted_predictions = self.convert_log_to_units(predictions)

            # Example prediction and logging results
            example_pred = self.model.predict(test_X.head(1))
            logging.info(f"Example prediction (log scale): {example_pred[0]}")
            logging.info(
                "Predicted UnitSales: " f"{self.convert_log_to_units(example_pred[0])}"
            )
            return converted_predictions


def main():
    inferencer = Inferencer(Config)
    test_df = load_processed_data(Config.PROCESSED_TEST_PATH)
    predictions = inferencer.get_predictions(test_df)
    return predictions


if __name__ == "__main__":
    main()
