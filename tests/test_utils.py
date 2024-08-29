import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from sklearn.ensemble import RandomForestRegressor
from app.config import Config
from app.utils import load_processed_data, evaluate_model, load_model, save_model

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config
        
        # Create a sample DataFrame
        self.sample_data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'UnitSales': np.random.randint(1, 100, 100),
            'DateKey': pd.date_range(start='2023-01-01', periods=100)
        })
        
        # Create a sample model
        self.sample_model = RandomForestRegressor(n_estimators=10, random_state=42)
        X = self.sample_data[['feature1', 'feature2']]
        y = self.sample_data['UnitSales']
        self.sample_model.fit(X, y)

    def test_load_processed_data(self):
        # Save sample data to a temporary file
        temp_file = os.path.join(self.temp_dir, 'test_data.csv')
        self.sample_data.to_csv(temp_file, index=False)
        
        # Test loading the data
        loaded_data = load_processed_data(temp_file)
        loaded_data['DateKey'] = pd.to_datetime(loaded_data['DateKey'])
        self.assertIsInstance(loaded_data, pd.DataFrame)
        self.assertEqual(loaded_data.shape, self.sample_data.shape)
        pd.testing.assert_frame_equal(loaded_data, self.sample_data)

    def test_load_processed_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_processed_data('non_existent_file.csv')

    def test_evaluate_model(self):
        X = self.sample_data[['feature1', 'feature2']]
        y = self.sample_data['UnitSales']
        
        rmse, mae = evaluate_model(self.sample_model, X, y)
        
        self.assertIsInstance(rmse, float)
        self.assertIsInstance(mae, float)
        self.assertGreater(rmse, 0)
        self.assertGreater(mae, 0)

    def test_load_and_save_model(self):
        # Save the model
        temp_model_file = os.path.join(self.temp_dir, 'test_model.pkl')
        save_model(self.sample_model, temp_model_file)
        
        # Check if the file exists
        self.assertTrue(os.path.exists(temp_model_file))
        
        # Load the model
        loaded_model = load_model(temp_model_file)
        
        # Check if the loaded model is of the same type
        self.assertIsInstance(loaded_model, RandomForestRegressor)
        
        # Compare predictions from original and loaded model
        X = self.sample_data[['feature1', 'feature2']]
        original_predictions = self.sample_model.predict(X)
        loaded_predictions = loaded_model.predict(X)
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)

    def test_load_model_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_model('non_existent_model.pkl')

    def test_save_model_directory_not_found(self):
        with self.assertRaises(FileNotFoundError):
            save_model(self.sample_model, '/non_existent_directory/model.pkl')

    def tearDown(self):
        # Clean up temporary directory
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

if __name__ == '__main__':
    unittest.main()