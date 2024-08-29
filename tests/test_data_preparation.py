import unittest
import pandas as pd
import numpy as np
from app.config import Config
from app.data_preparation import DataPreprocessor

class TestFeatureGeneration(unittest.TestCase):

    def setUp(self):
        self.config = Config
        self.preprocessor = DataPreprocessor(self.config)

        dates = pd.date_range(start='2023-01-01', end='2023-01-04')

        self.sample_df = pd.DataFrame({
            'DateKey': dates,
            'StoreCount': [100, 101, 102, 103],
            'ShelfCapacity': [1000, 1000, 1000, 1000],
            'PromoShelfCapacity': [50, 50, 50, 50],
            'IsPromo': [True, False, True, False],
            'ItemNumber': [1, 2, 1, 2],
            'CategoryCode': [100, 200, 100, 200],
            'GroupCode': [10, 20, 10, 20],
            'UnitSales': [100, 200, 300, 400]
        })

    def test_load_data(self):
        # Mock the pd.read_csv function
        pd.read_csv = lambda *args, **kwargs: self.sample_df
        
        loaded_df = self.preprocessor.load_data("dummy_path.csv")
        self.assertIsInstance(loaded_df, pd.DataFrame)
        self.assertEqual(len(loaded_df), len(self.sample_df))

    def test_add_time_features(self):
        df_with_time = self.preprocessor.add_time_features(self.sample_df)
        self.assertIn('month', df_with_time.columns)
        self.assertIn('weekday', df_with_time.columns)
        self.assertEqual(df_with_time['month'].dtype, 'category')
        self.assertEqual(df_with_time['weekday'].dtype, 'category')

    def test_convert_categorical(self):
        df_categorical = self.preprocessor.convert_categorical(self.sample_df)
        for col in ['GroupCode', 'ItemNumber', 'CategoryCode']:
            self.assertEqual(df_categorical[col].dtype, 'category')

    def test_preprocess_data(self):
        preprocessed_df = self.preprocessor.preprocess_data(self.sample_df)
        self.assertIn('month', preprocessed_df.columns)
        self.assertIn('weekday', preprocessed_df.columns)

    def test_train_test_split(self):
        train_df, test_df = self.preprocessor.train_test_split(self.sample_df, train_split_ratio=0.8)
        self.assertGreater(len(train_df), len(test_df))
        self.assertEqual(len(train_df) + len(test_df), len(self.sample_df))

    def test_add_lagged_features(self):
        df_with_lags = self.preprocessor.add_lagged_features(self.sample_df, 'UnitSales')
        for lag in self.config.LAGS:
            self.assertIn(f'UnitSales_lag_{lag}', df_with_lags.columns)

    def test_save_processed_data(self):
        # Mock the to_csv function
        pd.DataFrame.to_csv = lambda self, path, index: None
        
        try:
            self.preprocessor.save_processed_data(self.sample_df, self.sample_df)
        except Exception as e:
            self.fail(f"save_processed_data raised {type(e).__name__} unexpectedly!")

if __name__ == '__main__':
    unittest.main()