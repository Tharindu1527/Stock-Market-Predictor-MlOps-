# Stock-predictions-backend/ml/management/commands/train_models.py
from django.core.management.base import BaseCommand
from ml.model_registry import ModelRegistry
from ml.monitoring import ModelMonitor
from ml.config import config
import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import math
import logging
import time
import re

class Command(BaseCommand):
    help = 'Train new models for specified stocks'

    def add_arguments(self, parser):
        parser.add_argument('--stocks', type=str, help='Comma-separated list of stock symbols to train')
        parser.add_argument('--days', type=int, default=730, help='Number of days of historical data to use')
        parser.add_argument('--force', action='store_true', help='Force training even if recent model exists')
        parser.add_argument('--epochs', type=int, help='Override default epochs for training')
        parser.add_argument('--use-sample-data', action='store_true', help='Use generated sample data instead of real data')
        parser.add_argument('--batch-size', type=int, help='Override batch size for training')
        parser.add_argument('--min-data-points', type=int, default=120, help='Minimum number of data points required')

    def handle(self, *args, **options):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("model_training.log"),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger("model_training")
        
        # Get stocks to train
        stocks = options.get('stocks')
        if stocks:
            stocks = stocks.split(',')
        else:
            # Use default stocks from config
            stocks = config.get('api.default_stocks', ['AAPL', 'MSFT', 'NFLX', 'NVDA', 'DIS'])
        
        days = options.get('days', 730)
        force = options.get('force', False)
        use_sample_data = options.get('use_sample_data')
        min_data_points = options.get('min_data_points', 120)
        
        # Override epochs if specified
        epochs = options.get('epochs')
        if epochs is not None:
            training_epochs = epochs
        else:
            training_epochs = config.get('model.epochs', 1)
            
        # Override batch size if specified
        batch_size = options.get('batch_size')
        if batch_size is not None:
            training_batch_size = batch_size
        else:
            training_batch_size = config.get('model.batch_size', 1)
        
        logger.info(f"Starting training for stocks: {', '.join(stocks)}")
        
        # Initialize registry and monitor
        registry = ModelRegistry()
        monitor = ModelMonitor()
        
        # Train models for each stock
        trained_models = []
        
        for stock in stocks:
            try:
                # Check if we already have a recent model
                if not force:
                    existing_models = registry.list_models(stock=stock)
                    if existing_models:
                        latest_model = existing_models[-1]
                        try:
                            # Fix the timestamp parsing
                            timestamp = latest_model['timestamp']
                            # Convert from format like "20250401_161834" to "2025-04-01T16:18:34"
                            if '_' in timestamp:
                                date_part = timestamp[:8]
                                time_part = timestamp[9:]
                                formatted_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"
                                formatted_time = f"{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}"
                                model_date = datetime.fromisoformat(f"{formatted_date}T{formatted_time}")
                            else:
                                model_date = datetime.now() - timedelta(days=10)  # Default if parsing fails
                                
                            days_since_model = (datetime.now() - model_date).days
                            
                            if days_since_model < 7:  # Skip if model is less than a week old
                                logger.info(f"Skipping {stock}: Recent model exists from {model_date.strftime('%Y-%m-%d')}")
                                continue
                        except (ValueError, KeyError, IndexError) as e:
                            logger.warning(f"Error parsing timestamp for model: {e}, proceeding with training")
                
                logger.info(f"Training model for {stock} using {days} days of data")
                
                # Set up date range for data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Try to download real data if not using sample data
                df = pd.DataFrame()
                if not use_sample_data:
                    try:
                        logger.info(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                        df = yf.download(stock, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
                    except Exception as e:
                        logger.error(f"Error downloading data: {str(e)}")
                        df = pd.DataFrame()
                
                # Use sample data if real data is empty or if use_sample_data is True
                if df.empty:
                    if use_sample_data:
                        logger.info(f"Generating sample data for {stock}")
                    else:
                        logger.warning(f"No data available for {stock}, falling back to sample data")
                    
                    # Generate synthetic data for training - ensure sufficient data points
                    num_days = max(days, min_data_points)
                    date_range = pd.date_range(start=end_date - timedelta(days=num_days), end=end_date, freq='B')  # Business days
                    
                    # Generate synthetic stock data
                    np.random.seed(hash(stock) % 10000)  # Use stock name for reproducible randomness
                    initial_price = 100 + hash(stock) % 900  # Different starting price for each stock
                    prices = [initial_price]
                    
                    for i in range(1, len(date_range)):
                        # Random daily change between -2% and +2%
                        change = np.random.normal(0.0005, 0.015)
                        prices.append(prices[-1] * (1 + change))
                    
                    # Create DataFrame with synthetic data
                    df = pd.DataFrame({
                        'Open': prices,
                        'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
                        'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
                        'Close': prices,
                        'Adj Close': prices,
                        'Volume': [int(np.random.normal(1000000, 300000)) for _ in prices]
                    }, index=date_range)
                    
                    logger.info(f"Created synthetic data with {len(df)} records")
                else:
                    logger.info(f"Downloaded {len(df)} days of data")
                
                # Ensure we have enough data points
                if len(df) < min_data_points:
                    logger.warning(f"Not enough data points for {stock}. Got {len(df)}, need at least {min_data_points}. Generating more synthetic data.")
                    
                    # Add more synthetic data to reach minimum required
                    num_additional_days = min_data_points - len(df)
                    start_date_additional = df.index[0] - timedelta(days=num_additional_days * 1.5)  # Add extra buffer for weekends
                    
                    date_range_additional = pd.date_range(start=start_date_additional, end=df.index[0] - timedelta(days=1), freq='B')
                    
                    if len(date_range_additional) > 0:
                        # Get initial price close to the first real price
                        if df.empty:
                            initial_price_additional = 100 + hash(stock) % 900
                        else:
                            initial_price_additional = df['Close'].iloc[0] * 0.95  # Slightly lower than first real price
                        
                        prices_additional = [initial_price_additional]
                        
                        for i in range(1, len(date_range_additional)):
                            change = np.random.normal(0.0005, 0.015)
                            prices_additional.append(prices_additional[-1] * (1 + change))
                        
                        # Create DataFrame with additional synthetic data
                        df_additional = pd.DataFrame({
                            'Open': prices_additional,
                            'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices_additional],
                            'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices_additional],
                            'Close': prices_additional,
                            'Adj Close': prices_additional,
                            'Volume': [int(np.random.normal(1000000, 300000)) for _ in prices_additional]
                        }, index=date_range_additional)
                        
                        # Combine datasets
                        df = pd.concat([df_additional, df])
                        logger.info(f"Added {len(df_additional)} synthetic data points. Total records: {len(df)}")
                
                # Prepare data
                data = df['Close'].values.reshape(-1, 1)
                
                # Determine training size
                training_data_len = math.ceil(len(data) * config.get('model.training_split', 0.8))
                
                # Ensure we have enough data
                window_size = config.get('model.window_size', 60)
                if len(data) < window_size + 10:  # Need at least window_size + some extra data points
                    logger.error(f"Not enough data points for {stock} (need at least {window_size+10}, got {len(data)})")
                    continue
                
                # Scale data
                scaler = MinMaxScaler(feature_range=(0,1))
                scaled_data = scaler.fit_transform(data)
                
                # Create the training dataset
                train_data = scaled_data[0:training_data_len, :]
                x_train = []
                y_train = []
                
                for i in range(window_size, len(train_data)):
                    x_train.append(train_data[i-window_size:i, 0])
                    y_train.append(train_data[i, 0])
                
                # Convert to numpy arrays
                x_train, y_train = np.array(x_train), np.array(y_train)
                
                # Check if we have enough training data
                if len(x_train) == 0 or len(y_train) == 0:
                    logger.error(f"Not enough training data after preprocessing for {stock}")
                    continue
                
                # Reshape data for LSTM
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
                
                # Build model
                model = tf.keras.Sequential([
                    tf.keras.layers.LSTM(
                        config.get('model.lstm_units', 50), 
                        return_sequences=True, 
                        input_shape=(x_train.shape[1], 1)
                    ),
                    tf.keras.layers.LSTM(
                        config.get('model.lstm_units', 50), 
                        return_sequences=False
                    ),
                    tf.keras.layers.Dense(config.get('model.dense_units', 25)),
                    tf.keras.layers.Dense(1)
                ])
                
                # Compile model
                model.compile(
                    optimizer=config.get('model.optimizer', 'adam'),
                    loss='mean_squared_error'
                )
                
                logger.info(f"Training model with {len(x_train)} samples, {training_epochs} epochs")
                
                # Record start time
                start_time = time.time()
                
                # Train model
                model.fit(
                    x_train, 
                    y_train, 
                    batch_size=training_batch_size,
                    epochs=training_epochs,
                    verbose=1
                )
                
                # Calculate training time
                training_time = time.time() - start_time
                
                # Create test data set
                test_data = scaled_data[training_data_len - window_size:, :]
                x_test = []
                y_test = data[training_data_len:, :]
                
                for i in range(window_size, len(test_data)):
                    x_test.append(test_data[i-window_size:i, 0])
                
                # Convert to numpy array
                x_test = np.array(x_test)
                
                # Check if there's enough test data
                if len(x_test) == 0:
                    logger.error(f"Not enough test data after preprocessing for {stock}")
                    continue
                
                # Reshape data
                x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
                
                # Get predictions
                predictions = model.predict(x_test)
                predictions = scaler.inverse_transform(predictions)
                
                # Calculate RMSE
                rmse = np.sqrt(np.mean((predictions - y_test)**2))
                mae = np.mean(np.abs(predictions - y_test))
                
                logger.info(f"Model trained in {training_time:.2f}s with RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                
                # Save model to registry
                model_id = registry.save_model(
                    model=model,
                    scaler=scaler,
                    metadata={
                        'stock': stock,
                        'start_date': start_date.strftime('%Y-%m-%d'),
                        'end_date': end_date.strftime('%Y-%m-%d'),
                        'training_data_len': training_data_len,
                        'window_size': window_size,
                        'epochs': training_epochs,
                        'batch_size': training_batch_size,
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'training_time': training_time,
                        'performance_metrics': {
                            'rmse': float(rmse),
                            'mae': float(mae)
                        },
                        'is_sample_data': use_sample_data or df.empty,
                        'data_points': len(df)
                    }
                )
                
                logger.info(f"Model {model_id} saved to registry")
                trained_models.append(model_id)
                
                # Log model performance
                monitor.log_model_performance(
                    model_id=model_id,
                    stock=stock,
                    metrics={
                        "rmse": float(rmse),
                        "mae": float(mae),
                        "training_time": training_time
                    }
                )
                
                self.stdout.write(self.style.SUCCESS(f"Successfully trained model for {stock}"))
                
            except Exception as e:
                logger.error(f"Error training model for {stock}: {str(e)}", exc_info=True)
                self.stderr.write(self.style.ERROR(f"Error training model for {stock}: {str(e)}"))
                continue
        
        logger.info("Training complete")
        return trained_models  # Return model IDs for testing purposes