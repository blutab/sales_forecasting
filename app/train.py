import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
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
        self.client = MlflowClient()

    def train_model(
        self, train_X: pd.DataFrame, train_y: pd.Series, model_params: dict
    ) -> RandomForestRegressor:
        """
        Train a RandomForestRegressor model.

        Parameters:
        - train_X: pd.DataFrame : Features for training.
        - train_y: pd.Series : Target variable for training.
        - model_params: dict : Parameters for the RandomForestRegressor.

        Returns:
        - RandomForestRegressor : Trained model.
        """
        logging.info(f"Training RandomForestRegressor model with {model_params}")

        model = RandomForestRegressor(**model_params)
        model.fit(train_X, train_y)
        logging.info("Model training completed")
        return model

    def register_model(self, run_id: str, model_name: str) -> str:
        """
        Register the model in the MLflow Model Registry.

        Parameters:
        - run_id: str : The ID of the MLflow run that produced the model.
        - model_name: str : The name to give the model in the registry.

        Returns:
        - str : The version number of the registered model.
        """
        model_uri = f"runs:/{run_id}/model"
        model_details = mlflow.register_model(model_uri, model_name)
        return model_details.version

    def transition_model_stage(self, model_name: str, version: str, stage: str):
        """
        Transition a model to a new stage in the Model Registry.

        Parameters:
        - model_name: str : The name of the model in the registry.
        - version: str : The version of the model to transition.
        - stage: str : The new stage for the model (e.g., "Staging", "Production").
        """
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )

    def run_training(self):
        with mlflow.start_run() as run:
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
            predictions, rmse, mae = evaluate_model(model, train_X, train_y)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.sklearn.log_model(model, "model")

            save_model(model, self.config.MODEL_PATH)

            logging.info(f"Model trained and saved to {self.config.MODEL_PATH}")

            # Register the model
            model_name = "RandomForestRegressor"
            model_version = self.register_model(run.info.run_id, model_name)
            logging.info(f"Model registered with name: {model_name}, version: {model_version}")

            # Transition the model to staging
            self.transition_model_stage(model_name, model_version, "Staging")
            logging.info(f"Model version {model_version} transitioned to Staging")

            # TODO: Add logic to compare with production model and transition to production if better
            # For now, we'll just log a message
            logging.info("TODO: Implement comparison with production model")

def main():
    trainer = ModelTrainer(Config)
    trainer.run_training()

if __name__ == "__main__":
    main()