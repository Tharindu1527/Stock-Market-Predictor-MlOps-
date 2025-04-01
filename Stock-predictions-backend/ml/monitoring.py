# Stock-predictions-backend/ml/monitoring.py
import os
import json
import time
import logging
import pandas as pd
from datetime import datetime

class ModelMonitor:
    def __init__(self, log_dir="monitoring_logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Set up logging
        self.logger = logging.getLogger("model_monitoring")
        self.logger.setLevel(logging.INFO)
        
        # Create handler for the monitoring log file
        log_file = os.path.join(log_dir, f"monitoring_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Prediction log file paths
        self.prediction_logs_dir = os.path.join(log_dir, "predictions")
        os.makedirs(self.prediction_logs_dir, exist_ok=True)
        
        # Model performance logs
        self.performance_logs_dir = os.path.join(log_dir, "performance")
        os.makedirs(self.performance_logs_dir, exist_ok=True)
        
    def log_prediction(self, stock, prediction_value, actual_value=None, model_id=None, execution_time=None, error=None):
        """Log each prediction event"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "stock": stock,
            "model_id": model_id,
            "prediction_value": float(prediction_value) if prediction_value is not None else None,
            "actual_value": float(actual_value) if actual_value is not None else None,
            "execution_time_ms": execution_time,
            "error": error
        }
        
        # Log to prediction file
        log_file = os.path.join(self.prediction_logs_dir, f"predictions_{datetime.now().strftime('%Y%m%d')}.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
        
        # Log to general logger
        if error:
            self.logger.error(f"Prediction error for {stock}: {error}")
        else:
            self.logger.info(f"Prediction for {stock}: {prediction_value}")
    
    def log_model_performance(self, model_id, stock, metrics):
        """Log model performance metrics"""
        timestamp = datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "model_id": model_id,
            "stock": stock,
            "metrics": metrics
        }
        
        # Log to performance file
        log_file = os.path.join(self.performance_logs_dir, f"model_performance_{stock}.jsonl")
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + "\n")
        
        self.logger.info(f"Performance logged for model {model_id}, stock {stock}")
    
    def get_prediction_history(self, stock=None, limit=100):
        """Get prediction history, optionally filtered by stock"""
        all_predictions = []
        
        # Get all prediction files
        for filename in os.listdir(self.prediction_logs_dir):
            if filename.startswith("predictions_") and filename.endswith(".jsonl"):
                file_path = os.path.join(self.prediction_logs_dir, filename)
                with open(file_path, 'r') as f:
                    for line in f:
                        prediction = json.loads(line.strip())
                        if stock is None or prediction.get("stock") == stock:
                            all_predictions.append(prediction)
        
        # Sort by timestamp
        all_predictions.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Apply limit
        if limit > 0 and len(all_predictions) > limit:
            all_predictions = all_predictions[:limit]
        
        return all_predictions