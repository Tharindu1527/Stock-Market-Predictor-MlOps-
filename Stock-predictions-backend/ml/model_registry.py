# Stock-predictions-backend/ml/model_registry.py
import os
import json
import datetime
import tensorflow as tf
import pickle
import numpy as np

class ModelRegistry:
    def __init__(self, registry_dir="model_registry"):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)
        self.metadata_file = os.path.join(registry_dir, "registry_index.json")
        if not os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'w') as f:
                json.dump({"models": []}, f)
    
    def save_model(self, model, scaler, metadata):
        """Save model and its metadata"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_id = f"{metadata.get('stock', 'unknown')}_{timestamp}"
        model_dir = os.path.join(self.registry_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, "model.h5")
        try:
            model.save(model_path)
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            # Try saving in TensorFlow SavedModel format instead
            try:
                model_path = os.path.join(model_dir, "model")
                model.save(model_path)
            except Exception as e:
                print(f"Error saving model in SavedModel format: {str(e)}")
                return None
        
        # Save scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Prepare metadata
        full_metadata = {
            "model_id": model_id,
            "timestamp": timestamp,
            "stock": metadata.get('stock', 'unknown'),
            "start_date": metadata.get('start_date', ''),
            "end_date": metadata.get('end_date', ''),
            "training_data_len": metadata.get('training_data_len', 0),
            "window_size": metadata.get('window_size', 60),
            "rmse": metadata.get('rmse', float('inf')),
            "mae": metadata.get('mae', float('inf')),
            "performance_metrics": metadata.get('performance_metrics', {}),
            "model_path": model_path,
            "scaler_path": scaler_path,
            "status": "active",
            "is_sample_data": metadata.get('is_sample_data', False)
        }
        
        # Save individual metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f)
        
        # Update registry index
        try:
            with open(self.metadata_file, 'r') as f:
                registry_data = json.load(f)
        except json.JSONDecodeError:
            # If the registry file is corrupted, create a new one
            registry_data = {"models": []}
        
        registry_data["models"].append(full_metadata)
        
        with open(self.metadata_file, 'w') as f:
            json.dump(registry_data, f)
            
        return model_id
    
    def load_model(self, model_id=None, stock=None):
        """Load a model by ID or get latest for a stock"""
        if not model_id and not stock:
            # Get the latest model
            models = self.list_models()
            if not models:
                return None, None, None
            model_id = models[-1]["model_id"]
        elif stock and not model_id:
            # Get the latest model for the specific stock
            models = self.list_models(stock=stock)
            if not models:
                return None, None, None
            model_id = models[-1]["model_id"]
        
        # Find the model directory
        model_dir = os.path.join(self.registry_dir, model_id)
        if not os.path.exists(model_dir):
            return None, None, None
        
        # Load metadata
        metadata_path = os.path.join(model_dir, "metadata.json")
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading metadata: {str(e)}")
            return None, None, None
        
        # Load model
        try:
            model_path = metadata.get("model_path", os.path.join(model_dir, "model.h5"))
            if os.path.isdir(model_path):
                # This is a SavedModel directory
                model = tf.keras.models.load_model(model_path)
            else:
                # This is an H5 file
                model = tf.keras.models.load_model(model_path)
        except Exception as e:
            print(f"Error loading model from {model_path}: {str(e)}")
            return None, None, None
        
        # Load scaler
        try:
            scaler_path = metadata.get("scaler_path", os.path.join(model_dir, "scaler.pkl"))
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        except Exception as e:
            print(f"Error loading scaler: {str(e)}")
            return None, None, None
        
        return model, scaler, metadata
    
    def list_models(self, stock=None, limit=10):
        """List available models, optionally filtered by stock"""
        if not os.path.exists(self.metadata_file):
            return []
        
        try:
            with open(self.metadata_file, 'r') as f:
                registry_data = json.load(f)
            
            models = registry_data.get("models", [])
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading registry file: {str(e)}")
            return []
        
        # Filter by stock if specified
        if stock:
            models = [m for m in models if m.get("stock") == stock]
        
        # Sort by timestamp
        models.sort(key=lambda x: x.get('timestamp', ''))
        
        # Apply limit
        if limit > 0 and len(models) > limit:
            models = models[-limit:]
            
        return models
    
    def get_best_model(self, stock, metric="rmse", n_days=30):
        """Get the best performing model for a stock based on the metric"""
        models = self.list_models(stock=stock)
        if not models:
            return None, None, None
        
        if metric == "rmse" or metric == "mae":
            # Lower is better for RMSE and MAE
            best_model = min(models, key=lambda x: x.get(metric, float('inf')))
        else:
            # Assume higher is better for other metrics
            best_model = max(models, key=lambda x: x.get('performance_metrics', {}).get(metric, 0))
        
        return self.load_model(best_model["model_id"])