import pandas as pd
import numpy as np
import datetime
import logging
from typing import Tuple, List
from app import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(path: str) -> pd.DataFrame:
    """
    Load the dataset from the given path and perform initial preprocessing.

    Parameters:
    - path: str : Path to the dataset file.

    Returns:
    - pd.DataFrame : Preprocessed DataFrame.
    """
    logging.info(f"Loading data from {path}")
    df = pd.read_csv(path, sep=";", header=0)
    df['UnitSales'] = np.log(df['UnitSales'])
    df['DateKey'] = pd.to_datetime(df['DateKey'], format='%Y%m%d')
    df['month'] = df['DateKey'].dt.month.astype('category')
    df['weekday'] = df['DateKey'].dt.weekday.astype('category')
    df['GroupCode'] = df['GroupCode'].astype('category')
    df['ItemNumber'] = df['ItemNumber'].astype('category')
    df['CategoryCode'] = df['CategoryCode'].astype('category')

    return df.dropna(subset=['UnitSales', 'ShelfCapacity'])

def train_test_split(df: pd.DataFrame, train_split_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and test sets based on a date split.

    Parameters:
    - df: pd.DataFrame : The DataFrame to be split.
    - train_split_ratio: float : The ratio of the data to use for training.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame] : The training and test DataFrames.
    """
    split_date = df['DateKey'].quantile(train_split_ratio).date()
    train_df = df[df['DateKey'].dt.date <= split_date]
    test_df = df[df['DateKey'].dt.date > split_date]
    return train_df, test_df

def add_lagged_features(df: pd.DataFrame, lags: List[int], feature: str) -> pd.DataFrame:
    """
    Add lagged features to the DataFrame.

    Parameters:
    - df: pd.DataFrame : The DataFrame to add lagged features to.
    - lags: List[int] : A list of lag intervals.
    - feature: str : The feature to create lags for.

    Returns:
    - pd.DataFrame : DataFrame with lagged features added.
    """
    logging.info(f"Adding lagged features for {feature} with lags {lags}")

    # Ensure the DataFrame is sorted by 'ItemNumber' and 'DateKey'
    df = df.sort_values(by=['ItemNumber', 'DateKey'])
    
    for lag in lags:
        df[f'{feature}_lag_{lag}'] = df.groupby('ItemNumber')[feature].shift(lag)

    return df.dropna()

def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """
    Save the processed training and test data to CSV files.

    Parameters:
    - train_df: pd.DataFrame : The training DataFrame to be saved.
    - test_df: pd.DataFrame : The test DataFrame to be saved.
    """
    logging.info(f"Saving processed training data to {config.PROCESSED_TRAIN_PATH}")
    train_df.to_csv(config.PROCESSED_TRAIN_PATH, index=False)
    logging.info(f"Saving processed test data to {config.PROCESSED_TEST_PATH}")
    test_df.to_csv(config.PROCESSED_TEST_PATH, index=False)

def main():
    """
    Main function to orchestrate data loading, processing, and saving.
    """
    df = load_data(config.DATA_PATH)
    train_df, test_df = train_test_split(df)
    
    # You can parameterize range_of_lags if needed
    range_of_lags = [7, 14, 21]
    train_df_lag = add_lagged_features(train_df, range_of_lags, 'UnitSales')
    test_df_lag = add_lagged_features(test_df, range_of_lags, 'UnitSales')
    
    save_processed_data(train_df_lag, test_df_lag)
    logging.info(f"Processed data saved successfully.")

if __name__ == '__main__':
    main()