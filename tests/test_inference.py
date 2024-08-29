import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestRegressor
from app.config import Config
from app.inference import Inferencer, main
from app.utils import load_model, evaluate_model

class TestModelInference(unittest.TestCase):

    def setUp(self):
        self.config = Config

        with patch('app.utils.load_model') as mock_load_model:
            mock_model = MagicMock(spec=RandomForestRegressor)
            mock_load_model.return_value = mock_model
            self.inferencer = Inferencer(self.config)

    def test_convert_log_to_units(self):
        log_value = 5.0
        expected_units = round(np.exp(log_value))
        self.assertEqual(self.inferencer.convert_log_to_units(log_value), expected_units)
    
    def test_convert_log_to_units_edge_cases(self):
        self.assertEqual(self.inferencer.convert_log_to_units(0), 1)  # log(1) = 0
        self.assertEqual(self.inferencer.convert_log_to_units(-1), 0)  # Should round down to 0
        large_value = 100
        self.assertGreater(self.inferencer.convert_log_to_units(large_value), 1e40)  # Should be a very large number

if __name__ == '__main__':
    unittest.main()