import pandas as pd
import math
import mlflow
import logging
from app.config import Config
from app.utils import load_model, evaluate_model, load_processed_data

# Set up logging
logging.basicConfig(
    level=Config.LOGGING_LEVEL,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class Inferencer:
    def __init__(self, config: Config):
        self.config = config
        self.model = load_model(Config.MODEL_PATH)

    def convert_log_to_units(self, prediction: float) -> int:
        return round(math.exp(prediction))

    def run_inference(self, test_df: pd.DataFrame):
        with mlflow.start_run():
            # Separate features and target variable
            test_y = test_df["UnitSales"]
            test_X = test_df.drop(columns=["UnitSales", "DateKey"])

            # Evaluate the model
            rmse, mae = evaluate_model(self.model, test_X, test_y)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)

            # Example prediction and logging results
            example_pred = self.model.predict(test_X.head(1))
            logging.info(f"Example prediction (log scale): {example_pred[0]}")
            logging.info(
                "Predicted UnitSales: "
                f"{self.convert_log_to_units(example_pred[0])}"
            )


def main():
    inferencer = Inferencer(Config)
    test_df = load_processed_data(Config.PROCESSED_TEST_PATH)
    inferencer.run_inference(test_df)


if __name__ == "__main__":
    main()
