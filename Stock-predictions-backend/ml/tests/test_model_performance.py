from django.test import TestCase
from ml.model_registry import ModelRegistry
from ml.backtesting import ModelBacktester
import numpy as np
import os
import logging

class ModelPerformanceTest(TestCase):
    def setUp(self):
        # Set up directories
        os.makedirs("model_registry", exist_ok=True)
        os.makedirs("monitoring_logs", exist_ok=True)
        os.makedirs("backtest_results", exist_ok=True)
        
        # Set up logging to capture any issues
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("model_test")
    
    def test_model_training_and_evaluation(self):
        """Test that we can train a model and it performs reasonably well"""
        self.logger.info("Starting model training and evaluation test")
        
        import shutil
        if os.path.exists("model_registry"):
            self.logger.info("Clearing existing model registry")
            shutil.rmtree("model_registry")
        os.makedirs("model_registry", exist_ok=True)

        # Import the call_command function
        from django.core.management import call_command
        
        # Training with sample data and sufficient data points
        # The key change is adding force=True to ensure model training happens
        self.logger.info("Training model with sample data")
        call_command(
            'train_models', 
            stocks='AAPL', 
            use_sample_data=True, 
            force=True,  # Force training even if a recent model exists
            epochs=1, 
            days=150,
            min_data_points=150,
            batch_size=32
        )
        
        # Check that a model was created
        registry = ModelRegistry()
        models = registry.list_models(stock='AAPL')
        self.assertTrue(len(models) > 0, "No model was created")
        
        # Get the latest model
        latest_model = models[-1]
        self.logger.info(f"Created model: {latest_model['model_id']}")
        
        # Backtest with shorter test period and sample data
        backtester = ModelBacktester()
        self.logger.info("Running backtest with sample data")
        report = backtester.backtest_stock(
            'AAPL', 
            test_period_days=15,  # Shorter test period
            use_sample_data=True
        )
        
        # Check results
        self.assertIsNotNone(report, "Backtest failed to produce a report")
        
        if report:
            self.logger.info(f"Backtest metrics: RMSE={report['metrics']['rmse']:.4f}, MAE={report['metrics']['mae']:.4f}")
            # Check reasonable performance (adjust threshold as needed)
            self.assertLess(report['metrics']['rmse'], 50, "RMSE is too high")
            
            # Verify the report contains expected fields
            required_fields = ['stock', 'model_id', 'metrics', 'results_file', 'plot_file']
            for field in required_fields:
                self.assertIn(field, report, f"Report missing required field: {field}")
            
            # Verify the files exist
            self.assertTrue(os.path.exists(report['results_file']), "Results CSV file not created")
            self.assertTrue(os.path.exists(report['plot_file']), "Plot file not created")
            
    def tearDown(self):
        # Clean up test artifacts if needed
        self.logger.info("Test complete, cleaning up test artifacts")