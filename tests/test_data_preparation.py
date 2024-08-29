import unittest
import pandas as pd
from app.data_preparation import load_data, train_test_split, add_lagged_features

class TestFeatureGeneration(unittest.TestCase):

    def setUp(self):
        self.test_data = pd.DataFrame({
            'DateKey': ['20230101', '20230102', '20230103', '20230104'],
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
        # Test if load_data works correctly
        df = load_data('data/dataset.csv') 
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue('UnitSales' in df.columns)

    def test_train_test_split(self):
        # Test train_test_split functionality
        self.test_data['DateKey'] = pd.to_datetime(self.test_data['DateKey'], format='%Y%m%d')
        
        train_df, test_df = train_test_split(self.test_data, 0.75)
        self.assertEqual(len(train_df), 3)
        self.assertEqual(len(test_df), 1)

    def test_add_lagged_features(self):
        # Test adding lagged features
        df_with_lags = add_lagged_features(self.test_data, [1], 'UnitSales')
        self.assertIn('UnitSales_lag_1', df_with_lags.columns)

    def test_save_processed_data(self):
        pass 

if __name__ == '__main__':
    unittest.main()