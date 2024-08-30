import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import logging
from app.config import Config
from app.utils import load_processed_data, evaluate_model, save_model

# Set up logging
logging.basicConfig(
    level=Config.LOGGING_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config

    def train_model(
        self, train_X: pd.DataFrame, train_y: pd.Series, model_params: dict
    ) -> RandomForestRegressor:
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
        logging.info(f"Training RandomForestRegressor model with {model_params}")

        model = RandomForestRegressor(**model_params)
        model.fit(train_X, train_y)
        logging.info("Model training completed")
        return model

    def run_training(self):
        with mlflow.start_run():
            train_df = load_processed_data(self.config.PROCESSED_TRAIN_PATH)
            train_y = train_df["UnitSales"]
            train_X = train_df.drop(columns=["UnitSales", "DateKey"])

            MODEL_PARAMS = {
                "n_estimators": 100,
                "max_features": round(len(train_X.columns) / 3),
                "max_depth": len(train_X.columns),
                "random_state": 42,
            }

            mlflow.log_params(MODEL_PARAMS)

            model = self.train_model(train_X, train_y, MODEL_PARAMS)
            rmse, mae = evaluate_model(model, train_X, train_y)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.sklearn.log_model(model, "model")

            save_model(model, self.config.MODEL_PATH)

            logging.info(f"Model trained and saved to {Config.MODEL_PATH}")


def main():
    trainer = ModelTrainer(Config)
    trainer.run_training()


if __name__ == "__main__":
    main()
