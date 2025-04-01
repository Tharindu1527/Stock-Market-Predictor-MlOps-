# Stock-predictions-backend/ml/tests/test_model_performance.py
from django.test import TestCase
from ml.model_registry import ModelRegistry
from ml.backtesting import ModelBacktester
import numpy as np

class ModelPerformanceTest(TestCase):
    def test_model_training_and_evaluation(self):
        """Test that we can train a model and it performs reasonably well"""
        # This will use synthetic data by default
        from django.core.management import call_command
        
        # Train a model for AAPL with synthetic data - use at least 90 days
        # to ensure enough data for backtesting
        call_command('train_models', stocks='AAPL', use_sample_data=True, epochs=1, days=120)
        
        # Check that a model was created
        registry = ModelRegistry()
        models = registry.list_models(stock='AAPL')
        self.assertTrue(len(models) > 0, "No model was created")
        
        # Backtest the model with a smaller test period to ensure enough data
        backtester = ModelBacktester()
        report = backtester.backtest_stock('AAPL', test_period_days=20, use_sample_data=True)
        
        # Check that the model has reasonable performance
        self.assertIsNotNone(report, "Backtest failed to produce a report")
        if report:  # Only check metrics if report exists
            self.assertLess(report['metrics']['rmse'], 50, "RMSE is too high")