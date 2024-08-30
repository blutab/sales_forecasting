import unittest
import pandas as pd
import numpy as np
import os
import tempfile
from sklearn.ensemble import RandomForestRegressor
from unittest.mock import patch, MagicMock
import mlflow
from app.config import Config
from app.utils import load_processed_data, evaluate_model, load_model_from_registry, save_model

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config

        # Create a sample DataFrame
        self.sample_data = pd.DataFrame({
            'UnitSales': np.random.randint(1, 100, 100),
            'DateKey': pd.date_range(start='2023-01-01', periods=100),
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'feature3': np.random.rand(100),
            'feature4': np.random.rand(100),
            'feature5': np.random.rand(100),
            'feature6': np.random.rand(100),
            'feature7': np.random.rand(100)
        })

        # Create a sample model
        self.sample_model = RandomForestRegressor(n_estimators=10, random_state=42)
        X = self.sample_data.drop(['UnitSales', 'DateKey'], axis=1)
        y = self.sample_data['UnitSales']
        self.sample_model.fit(X, y)

    # def test_load_processed_data(self):
    #     # Save sample data to a temporary file
    #     temp_file = os.path.join(self.temp_dir, 'sample_data.csv')
    #     self.sample_data.to_csv(temp_file, index=False)

    #     # Test loading the data
    #     loaded_data = load_processed_data(temp_file)
    #     loaded_data['DateKey'] = pd.to_datetime(loaded_data['DateKey'])

    #     self.assertIsInstance(loaded_data, pd.DataFrame)
    #     self.assertEqual(loaded_data.shape, self.sample_data.shape)
    #     pd.testing.assert_frame_equal(loaded_data, self.sample_data)

    def test_load_processed_data_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            load_processed_data(os.path.join(self.temp_dir, 'non_existent_file.csv'))

    def test_evaluate_model(self):
        X = self.sample_data.drop(['UnitSales', 'DateKey'], axis=1)
        y = self.sample_data['UnitSales']

        predictions, rmse, mae = evaluate_model(self.sample_model, X, y)

        self.assertIsInstance(rmse, float)
        self.assertIsInstance(mae, float)
        self.assertGreater(rmse, 0)
        self.assertGreater(mae, 0)

    @patch('mlflow.pyfunc.load_model')
    def test_load_model_from_registry(self, mock_load_model):
        # Mock the MLflow model loading
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Test loading the model
        loaded_model = load_model_from_registry('test_model', 'Production')

        # Check if MLflow's load_model was called with the correct arguments
        mock_load_model.assert_called_once_with('models:/test_model/Production')

        # Check if the loaded model is the mocked model
        self.assertEqual(loaded_model, mock_model)

    @patch('mlflow.pyfunc.load_model')
    def test_load_model_from_registry_error(self, mock_load_model):
        # Mock MLflow raising an exception
        mock_load_model.side_effect = mlflow.exceptions.MlflowException("Model not found")

        # Test that the function raises the exception
        with self.assertRaises(mlflow.exceptions.MlflowException):
            load_model_from_registry('non_existent_model', 'Production')

    @patch('mlflow.sklearn.save_model')
    def test_save_model(self, mock_save_model):
        # Test saving the model
        temp_model_file = os.path.join(self.temp_dir, 'test_model')
        save_model(self.sample_model, temp_model_file)

        # Check if MLflow's save_model was called with the correct arguments
        mock_save_model.assert_called_once_with(self.sample_model, temp_model_file)

    @patch('mlflow.sklearn.save_model')
    def test_save_model_error(self, mock_save_model):
        # Mock MLflow raising an exception
        mock_save_model.side_effect = Exception("Error saving model")

        # Test that the function raises the exception
        with self.assertRaises(Exception):
            save_model(self.sample_model, '/non_existent_directory/model')

    def tearDown(self):
        # Clean up temporary directory
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

if __name__ == '__main__':
    unittest.main()