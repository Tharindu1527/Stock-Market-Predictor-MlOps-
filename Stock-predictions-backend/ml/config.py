# Stock-predictions-backend/ml/config.py
import os
import json

# Configuration class for managing settings
class MLConfig:
    def __init__(self, config_file=None):
        # Set default configuration
        self.config = {
            "model": {
                "lstm_units": 50,
                "dense_units": 25,
                "epochs": 1,
                "batch_size": 1,
                "window_size": 60,
                "training_split": 0.8,
                "optimizer": "adam"
            },
            "monitoring": {
                "enabled": True,
                "log_level": "INFO",
                "log_dir": "monitoring_logs"
            },
            "registry": {
                "enabled": True,
                "directory": "model_registry"
            },
            "api": {
                "enable_cache": False,
                "cache_timeout": 3600,
                "default_stocks": ["AAPL", "MSFT", "NFLX", "NVDA", "DIS"]
            }
        }
        
        # Load from file if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                self._update_dict(self.config, file_config)
        
        # Override with environment variables
        self._load_from_env()
    
    def _update_dict(self, target, source):
        """Recursively update a dictionary"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._update_dict(target[key], value)
            else:
                target[key] = value
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Model configuration
        if os.environ.get('MODEL_LSTM_UNITS'):
            self.config["model"]["lstm_units"] = int(os.environ.get('MODEL_LSTM_UNITS'))
        if os.environ.get('MODEL_DENSE_UNITS'):
            self.config["model"]["dense_units"] = int(os.environ.get('MODEL_DENSE_UNITS'))
        if os.environ.get('MODEL_EPOCHS'):
            self.config["model"]["epochs"] = int(os.environ.get('MODEL_EPOCHS'))
        if os.environ.get('MODEL_BATCH_SIZE'):
            self.config["model"]["batch_size"] = int(os.environ.get('MODEL_BATCH_SIZE'))
        if os.environ.get('MODEL_WINDOW_SIZE'):
            self.config["model"]["window_size"] = int(os.environ.get('MODEL_WINDOW_SIZE'))
        if os.environ.get('MODEL_TRAINING_SPLIT'):
            self.config["model"]["training_split"] = float(os.environ.get('MODEL_TRAINING_SPLIT'))
        if os.environ.get('MODEL_OPTIMIZER'):
            self.config["model"]["optimizer"] = os.environ.get('MODEL_OPTIMIZER')
        
        # Monitoring configuration
        if os.environ.get('MONITORING_ENABLED'):
            self.config["monitoring"]["enabled"] = os.environ.get('MONITORING_ENABLED').lower() == 'true'
        if os.environ.get('MONITORING_LOG_LEVEL'):
            self.config["monitoring"]["log_level"] = os.environ.get('MONITORING_LOG_LEVEL')
        if os.environ.get('MONITORING_LOG_DIR'):
            self.config["monitoring"]["log_dir"] = os.environ.get('MONITORING_LOG_DIR')
        
        # Registry configuration
        if os.environ.get('REGISTRY_ENABLED'):
            self.config["registry"]["enabled"] = os.environ.get('REGISTRY_ENABLED').lower() == 'true'
        if os.environ.get('REGISTRY_DIRECTORY'):
            self.config["registry"]["directory"] = os.environ.get('REGISTRY_DIRECTORY')
        
        # API configuration
        if os.environ.get('API_ENABLE_CACHE'):
            self.config["api"]["enable_cache"] = os.environ.get('API_ENABLE_CACHE').lower() == 'true'
        if os.environ.get('API_CACHE_TIMEOUT'):
            self.config["api"]["cache_timeout"] = int(os.environ.get('API_CACHE_TIMEOUT'))
        if os.environ.get('API_DEFAULT_STOCKS'):
            self.config["api"]["default_stocks"] = os.environ.get('API_DEFAULT_STOCKS').split(',')
    
    def get(self, key, default=None):
        """Get a configuration value with dot notation (e.g., 'model.lstm_units')"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def to_dict(self):
        """Return the full configuration as a dictionary"""
        return self.config.copy()

# Create a global config instance
config = MLConfig()