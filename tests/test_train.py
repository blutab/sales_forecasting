import unittest
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from app.train import load_processed_data, train_model

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        self.train_data = pd.DataFrame({
            'StoreCount': [100, 101, 102],
            'ShelfCapacity': [1000, 1000, 1000],
            'PromoShelfCapacity': [50, 50, 50],
            'IsPromo': [True, False, True],
            'ItemNumber': [1, 2, 1],
            'CategoryCode': [100, 200, 100],
            'GroupCode': [10, 20, 10],
            'month': [1, 1, 1],
            'weekday': [0, 1, 2],
            'UnitSales_lag_7': [100, 200, 300],
            'UnitSales_lag_14': [150, 250, 350],
            'UnitSales_lag_21': [200, 300, 400],
            'UnitSales': [1000, 2000, 3000]
        })

    def test_load_processed_data(self):
        # Test if load_processed_data works correctly
        df = load_processed_data('data/train_data.csv') 
        self.assertIsInstance(df, pd.DataFrame)

    def test_train_model(self):
        # Test if the model is trained correctly
        train_X = self.train_data.drop(columns=['UnitSales'])
        train_y = self.train_data['UnitSales']
        model = train_model(train_X, train_y)
        self.assertIsInstance(model, RandomForestRegressor)

    def test_save_model(self):
        pass 
    
if __name__ == '__main__':
    unittest.main()