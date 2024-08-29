import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.ensemble import RandomForestRegressor
from app.train import ModelTrainer
from app.config import Config
from app.utils import load_processed_data, evaluate_model, save_model

class TestTrainModel(unittest.TestCase):

    def setUp(self):
        self.config = Config
        self.model_trainer = ModelTrainer(self.config)


        self.sample_data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'UnitSales': np.random.randint(1, 100, 100),
            'DateKey': pd.date_range(start='2023-01-01', periods=100)
        })

    def test_train_model(self):
        train_X = self.sample_data.drop(columns=['UnitSales', 'DateKey'])
        train_y = self.sample_data['UnitSales']
        model_params = {
            'n_estimators': 10,
            'max_features': 1,
            'max_depth': 5,
            'random_state': 42
        }
        
        model = self.model_trainer.train_model(train_X, train_y, model_params)
        
        self.assertIsInstance(model, RandomForestRegressor)
        self.assertEqual(model.n_estimators, 10)
        self.assertEqual(model.max_features, 1)
        self.assertEqual(model.max_depth, 5)
        self.assertEqual(model.random_state, 42)

    def test_train_model_with_invalid_params(self):
        train_X = self.sample_data.drop(columns=['UnitSales', 'DateKey'])
        train_y = self.sample_data['UnitSales']
        invalid_params = {
            'n_estimators': 'invalid',
            'max_features': 'invalid',
            'max_depth': 'invalid',
            'random_state': 'invalid'
        }
        
        with self.assertRaises(TypeError):
            self.model_trainer.train_model(train_X, train_y, invalid_params)

    def test_train_model_with_empty_data(self):
        empty_X = pd.DataFrame()
        empty_y = pd.Series()
        model_params = {
            'n_estimators': 10,
            'max_features': 1,
            'max_depth': 5,
            'random_state': 42
        }
        
        with self.assertRaises(ValueError):
            self.model_trainer.train_model(empty_X, empty_y, model_params)

if __name__ == '__main__':
    unittest.main()