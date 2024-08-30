import pandas as pd
import numpy as np
import datetime
import logging
from typing import Tuple, List
from app.config import Config

logging.basicConfig(
    level=Config.LOGGING_LEVEL, format="%(asctime)s - %(levelname)s - %(message)s"
)


class DataPreprocessor:
    def __init__(self, config: Config):
        self.config = config

    def load_data(self, path: str) -> pd.DataFrame:
        """
        Load the dataset from the given path and perform initial preprocessing.

        Parameters:
        - path: str : Path to the dataset file.

        Returns:
        - pd.DataFrame : Preprocessed DataFrame.
        """
        logging.info(f"Loading data from {path}")
        df = pd.read_csv(path, sep=";", header=0)
        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-related features to the DataFrame.

        Parameters:
        - df: pd.DataFrame : Input DataFrame.

        Returns:
        - pd.DataFrame : DataFrame with time features added.
        """
        df["month"] = df["DateKey"].dt.month.astype("category")
        df["weekday"] = df["DateKey"].dt.weekday.astype("category")
        return df

    def convert_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert specific columns to categorical types.

        Parameters:
        - df: pd.DataFrame : Input DataFrame.

        Returns:
        - pd.DataFrame : DataFrame with categorical columns.
        """
        categorical_columns = ["GroupCode", "ItemNumber", "CategoryCode"]
        for col in categorical_columns:
            df[col] = df[col].astype("category")
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing steps to the DataFrame.

        Parameters:
        - df: pd.DataFrame : Input DataFrame.

        Returns:
        - pd.DataFrame : Preprocessed DataFrame.
        """
        df["UnitSales"] = np.log(df["UnitSales"])
        df["DateKey"] = pd.to_datetime(df["DateKey"], format="%Y%m%d")
        df = self.add_time_features(df)
        df = self.convert_categorical(df)
        return df.dropna()

    def train_test_split(
        self, df: pd.DataFrame, train_split_ratio: float = 0.8
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data into training and test sets based on a date split.

        Parameters:
        - df: pd.DataFrame : The DataFrame to be split.
        - train_split_ratio: float : The ratio of the data to use for training.

        Returns:
        - Tuple[pd.DataFrame, pd.DataFrame] : The training and test DataFrames.
        """
        split_date = df["DateKey"].quantile(train_split_ratio).date()
        train_df = df[df["DateKey"].dt.date <= split_date]
        test_df = df[df["DateKey"].dt.date > split_date]
        return train_df, test_df

    def add_lagged_features(self, df: pd.DataFrame, feature: str) -> pd.DataFrame:
        """
        Add lagged features to the DataFrame.

        Parameters:
        - df: pd.DataFrame : The DataFrame to add lagged features to.
        - lags: List[int] : A list of lag intervals.
        - feature: str : The feature to create lags for.

        Returns:
        - pd.DataFrame : DataFrame with lagged features added.
        """
        logging.info(f"Adding lagged features for {feature} with lags {Config.LAGS}")

        # Ensure the DataFrame is sorted by 'ItemNumber' and 'DateKey'
        df = df.sort_values(by=["ItemNumber", "DateKey"])

        for lag in Config.LAGS:
            df[f"{feature}_lag_{lag}"] = df.groupby("ItemNumber", observed=False)[
                feature
            ].shift(lag)

        return df.dropna()

    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Save the processed training and test data to CSV files.

        Parameters:
        - train_df: pd.DataFrame : The training DataFrame to be saved.
        - test_df: pd.DataFrame : The test DataFrame to be saved.
        """
        logging.info(f"Saving processed training data to {Config.PROCESSED_TRAIN_PATH}")
        train_df.to_csv(Config.PROCESSED_TRAIN_PATH, index=False)
        logging.info(f"Saving processed test data to {Config.PROCESSED_TEST_PATH}")
        test_df.to_csv(Config.PROCESSED_TEST_PATH, index=False)


def main():
    """
    Main function to orchestrate data loading, processing, and saving.
    """
    preprocessor = DataPreprocessor(Config)

    df = preprocessor.load_data(Config.DATA_PATH)
    df = preprocessor.preprocess_data(df)
    train_df, test_df = preprocessor.train_test_split(df)

    # You can parameterize range_of_lags if needed
    train_df_lag = preprocessor.add_lagged_features(train_df, "UnitSales")
    test_df_lag = preprocessor.add_lagged_features(test_df, "UnitSales")

    preprocessor.save_processed_data(train_df_lag, test_df_lag)
    logging.info(f"Processed data saved successfully.")


if __name__ == "__main__":
    main()
