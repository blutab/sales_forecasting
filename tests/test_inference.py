import unittest
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from app.inference import load_processed_data, load_model, evaluate_model, convert_log_to_units

class TestModelInference(unittest.TestCase):

    def setUp(self):
        self.test_data = pd.DataFrame({
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
        df = load_processed_data('data/test_data.csv')
        self.assertIsInstance(df, pd.DataFrame)

    def test_load_model(self):
        # Test if the model loads correctly
        model = load_model('models/forecasting_model.pkl')
        self.assertIsInstance(model, RandomForestRegressor)

    def test_evaluate_model(self):
        # Test if the evaluation works correctly
        model = RandomForestRegressor()
        model.fit(self.test_data.drop(columns=['UnitSales']), self.test_data['UnitSales'])
        rmse, mae = evaluate_model(model, self.test_data.drop(columns=['UnitSales']), self.test_data['UnitSales'])
        self.assertIsInstance(rmse, float)
        self.assertIsInstance(mae, float)

    def test_convert_log_to_units(self):
        # Test the conversion function
        log_value = 6.907755278982137  # log(1000)
        result = convert_log_to_units(log_value)
        self.assertEqual(result, 1000)

if __name__ == '__main__':
    unittest.main()