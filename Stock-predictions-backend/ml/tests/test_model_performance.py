# Stock-predictions-backend/ml/tests/test_model_performance.py
from django.test import TestCase
from django.core.management import call_command
from ml.model_registry import ModelRegistry
from ml.backtesting import ModelBacktester
import numpy as np
import os
import logging
import sys
import traceback

class ModelPerformanceTest(TestCase):
    def setUp(self):
        # Ensure necessary directories exist
        os.makedirs("model_registry", exist_ok=True)
        os.makedirs("monitoring_logs", exist_ok=True)
        os.makedirs("backtest_results", exist_ok=True)
        
        # Set up detailed logging
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("test_debug.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger("model_performance_test")
    
    def test_model_training_and_evaluation(self):
        """Comprehensive test for model training and backtesting"""
        try:
            self.logger.info("Starting comprehensive model performance test")
            
            # Training with verbose logging and sample data
            self.logger.info("Initiating model training")
            call_command(
                'train_models', 
                stocks='AAPL', 
                use_sample_data=True, 
                epochs=1, 
                days=150,
                min_data_points=150,
                batch_size=32,
                verbosity=2  # Increase verbosity for more detailed logs
            )
            
            # Check model registry
            registry = ModelRegistry()
            models = registry.list_models(stock='AAPL')
            self.assertTrue(len(models) > 0, "No model was created for AAPL")
            
            # Get the latest model details
            latest_model = models[-1]
            model_id = latest_model.get('model_id')
            self.logger.info(f"Latest model ID: {model_id}")
            
            # Attempt backtesting with detailed logging
            backtester = ModelBacktester()
            self.logger.info("Starting backtest")
            
            report = backtester.backtest_stock(
                'AAPL', 
                test_period_days=15,  
                use_sample_data=True
            )
            
            # Comprehensive assertions
            self.assertIsNotNone(report, "Backtest failed to produce a report")
            
            # Validate report structure
            required_keys = ['stock', 'model_id', 'metrics', 'results_file', 'plot_file']
            for key in required_keys:
                self.assertIn(key, report, f"Report missing required key: {key}")
            
            # Performance metric validations
            metrics = report.get('metrics', {})
            self.logger.info(f"Backtest Metrics: {metrics}")
            
            self.assertIn('rmse', metrics, "RMSE metric missing")
            self.assertIn('mae', metrics, "MAE metric missing")
            
            # Specific performance thresholds
            self.assertLess(metrics.get('rmse', float('inf')), 50, "RMSE is too high")
            
            # Verify output files exist
            self.assertTrue(os.path.exists(report.get('results_file', '')), 
                            "Results CSV file not created")
            self.assertTrue(os.path.exists(report.get('plot_file', '')), 
                            "Backtest plot file not created")
            
            self.logger.info("Model performance test completed successfully")
        
        except Exception as e:
            # Log full traceback for debugging
            self.logger.error(f"Test failed: {str(e)}")
            self.logger.error(traceback.format_exc())
            raise
    
    def tearDown(self):
        """Clean up after test"""
        self.logger.info("Cleaning up test artifacts")