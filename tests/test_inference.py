import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestRegressor
import mlflow
from app.config import Config
from app.inference import Inferencer, main
from app.utils import load_model_from_registry, evaluate_model

class TestModelInference(unittest.TestCase):

    def setUp(self):
        self.config = Config

        # Patch the load_model_from_registry function
        patcher = patch('app.inference.load_model_from_registry', autospec=True)
        self.mock_load_model_from_registry = patcher.start()

        # Ensure that the mock returns a MagicMock of a RandomForestRegressor
        self.mock_load_model_from_registry.return_value = MagicMock(spec=RandomForestRegressor)

        # Instantiate the Inferencer with the mocked model loading
        self.inferencer = Inferencer(self.config)

        # Stop patching after tests to clean up
        self.addCleanup(patcher.stop)

        # Patch mlflow.start_run to prevent actual MLflow runs during tests
        self.mlflow_patcher = patch('mlflow.start_run')
        self.mock_mlflow_start_run = self.mlflow_patcher.start()
        self.addCleanup(self.mlflow_patcher.stop)

    def test_convert_log_to_units(self):
        log_value = 5.0
        expected_units = round(np.exp(log_value))
        self.assertEqual(self.inferencer.convert_log_to_units(log_value), expected_units)

    def test_convert_log_to_units_edge_cases(self):
        self.assertEqual(self.inferencer.convert_log_to_units(0), 1)  # log(1) = 0
        self.assertEqual(self.inferencer.convert_log_to_units(-1), 0)  # Should round down to 0
        large_value = 100
        self.assertGreater(self.inferencer.convert_log_to_units(large_value), 1e40)  # Should be a very large number

    @patch('app.inference.evaluate_model')
    def test_get_predictions(self, mock_evaluate_model):
        # Create a sample DataFrame
        test_df = pd.DataFrame({
            'UnitSales': [100, 200, 300],
            'DateKey': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Feature1': [1, 2, 3],
            'Feature2': [4, 5, 6]
        })

        # Mock the evaluate_model function
        mock_evaluate_model.return_value = (np.array([4.5, 5.0, 5.5]), 0.1, 0.2)

        # Call get_predictions
        predictions = self.inferencer.get_predictions(test_df)

        # Assert that evaluate_model was called
        mock_evaluate_model.assert_called_once()

        # Check the shape of the predictions
        self.assertEqual(len(predictions), 3)

        # Check if the predictions are converted from log scale
        expected_predictions = np.round(np.exp(np.array([4.5, 5.0, 5.5]))).astype(int)
        np.testing.assert_array_equal(predictions, expected_predictions)

        # Check if mlflow.start_run was called
        self.mock_mlflow_start_run.assert_called_once()

    @patch('app.inference.logging.info')
    @patch('app.inference.evaluate_model')
    def test_logging_in_get_predictions(self, mock_evaluate_model, mock_logging):
        # Create a sample DataFrame with one row
        test_df = pd.DataFrame({
            'UnitSales': [100],
            'DateKey': ['2023-01-01'],
            'Feature1': [1],
            'Feature2': [4]
        })

        # Mock the evaluate_model function
        mock_evaluate_model.return_value = (np.array([4.5]), 0.1, 0.2)

        # Call get_predictions
        self.inferencer.get_predictions(test_df)

        # Assert that logging.info was called
        mock_logging.assert_called()

        # Check if mlflow.start_run was called
        self.mock_mlflow_start_run.assert_called_once()

    @patch('app.inference.Inferencer')
    @patch('app.inference.load_processed_data')
    def test_main(self, mock_load_processed_data, mock_inferencer):
        # Create a mock DataFrame
        mock_df = pd.DataFrame({'UnitSales': [100, 200], 'DateKey': ['2023-01-01', '2023-01-02'], 'Feature1': [1, 2]})
        mock_load_processed_data.return_value = mock_df

        # Create a mock Inferencer instance
        mock_inferencer_instance = MagicMock()
        mock_inferencer_instance.get_predictions.return_value = np.array([150, 250])
        mock_inferencer.return_value = mock_inferencer_instance

        # Call the main function
        result = main()

        # Assert that load_processed_data was called with the correct argument
        mock_load_processed_data.assert_called_once_with(Config.PROCESSED_TEST_PATH)

        # Assert that get_predictions was called with the mock DataFrame
        mock_inferencer_instance.get_predictions.assert_called_once_with(mock_df)

        # Check the result
        np.testing.assert_array_equal(result, np.array([150, 250]))

    def test_model_loading(self):
        # Test that the model is loaded correctly from the registry
        self.mock_load_model_from_registry.assert_called_once_with(
            Config.MODEL_NAME, Config.MODEL_STAGE
        )
        self.assertIsInstance(self.inferencer.model, MagicMock)
        self.assertIsInstance(self.inferencer.model, RandomForestRegressor)
        
if __name__ == '__main__':
    unittest.main()