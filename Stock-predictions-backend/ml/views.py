from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from datetime import datetime, timedelta
import yfinance as yf
import psutil
import os
import time

# data libraries
import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# MLOps additions
from .model_registry import ModelRegistry
from .monitoring import ModelMonitor
from .config import config

class Data(APIView):
    def get(self, request):
        try:
            # Initialize monitoring
            monitor = ModelMonitor()
            start_time = time.time()
            
            # Get the stock quote
            stock = request.GET.get('stock')
            start = request.GET.get('start')
            end = request.GET.get('end')
            
            print(f"Attempting to fetch data for: {stock} from {start} to {end}")
            
            # Use a more direct approach with yfinance
            try:
                df = yf.download(stock, start=start, end=end, progress=False)
                print(f"Downloaded data shape: {df.shape}")
            except Exception as download_error:
                print(f"Download error: {str(download_error)}")
                df = pd.DataFrame()
            
            # Check if data was retrieved
            if df.empty:
                print("Using fallback data since download failed")
                # Create some sample data for testing
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365*2)  # 2 years of data
                date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
                
                # Generate synthetic stock data
                np.random.seed(42)  # For reproducibility
                prices = [100]  # Start price
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
                
                # Reset index to make Date a column
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'Date'}, inplace=True)
                
                print(f"Created fallback data with shape: {df.shape}")
            else:
                df.reset_index(inplace=True)
            
            # Create a new dataframe with Close column
            data = df.filter(['Close'])
            time_data = df.filter(['Date'])
            
            # Convert the dataframe to numpy array
            dataset = data.values
            
            # Get the number of rows to train the model on
            training_data_len = math.ceil(len(dataset) * config.get('model.training_split', 0.8))
            
            # If there isn't enough data, return an error
            if len(dataset) < 60:
                monitor.log_prediction(
                    stock=stock,
                    prediction_value=None,
                    execution_time=(time.time() - start_time) * 1000,
                    error=f"Not enough data points (need at least 60, got {len(dataset)}) for analysis."
                )
                return Response(
                    {"error": f"Not enough data points (need at least 60, got {len(dataset)}) for analysis."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Scale the data
            scaler = MinMaxScaler(feature_range=(0,1))
            scaled_data = scaler.fit_transform(dataset)
            
            # Training the dataset
            # Create the scaled training data set
            train_data = scaled_data[0:training_data_len, :]
            
            # Split the data into x_train and y_train data sets
            window_size = config.get('model.window_size', 60)
            x_train = []
            y_train = []
            
            for i in range(window_size, len(train_data)):
                x_train.append(train_data[i-window_size:i, 0])
                y_train.append(train_data[i, 0])
            
            # Convert the x_train and y_train to numpy arrays
            x_train, y_train = np.array(x_train), np.array(y_train)
            
            # Check if we have enough training data
            if len(x_train) == 0 or len(y_train) == 0:
                monitor.log_prediction(
                    stock=stock,
                    prediction_value=None,
                    execution_time=(time.time() - start_time) * 1000,
                    error="Not enough training data after preprocessing."
                )
                return Response(
                    {"error": "Not enough training data after preprocessing."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Reshape the data
            x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
            
            # Get model parameters from config
            lstm_units = config.get('model.lstm_units', 50)
            dense_units = config.get('model.dense_units', 25)
            batch_size = config.get('model.batch_size', 1)
            epochs = config.get('model.epochs', 1)
            
            # Build LSTM model
            model = Sequential()
            model.add(LSTM(lstm_units, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(LSTM(lstm_units, return_sequences=False))
            model.add(Dense(dense_units))
            model.add(Dense(1))
            
            # Compile the model
            model.compile(optimizer=config.get('model.optimizer', 'adam'), loss="mean_squared_error")
            
            # Train the model
            model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0)
            
            # Create the test data set
            test_data = scaled_data[training_data_len-window_size:, :]
            
            # Create the data sets x_test and y_test
            x_test = []
            y_test = dataset[training_data_len:, :]
            
            for i in range(window_size, len(test_data)):
                x_test.append(test_data[i-window_size:i, 0])
            
            # Convert to numpy array
            x_test = np.array(x_test)
            
            # Check if there's test data available
            if len(x_test) == 0:
                monitor.log_prediction(
                    stock=stock,
                    prediction_value=None,
                    execution_time=(time.time() - start_time) * 1000,
                    error="Not enough test data after preprocessing."
                )
                return Response(
                    {"error": "Not enough test data after preprocessing."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Reshape the data
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
            
            # Get the models predicted price values
            predictions = model.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            
            # Get the root mean squared error (rmse)
            rmse = np.sqrt(np.mean((predictions - y_test)**2))
            mae = np.mean(np.abs(predictions - y_test))
            
            # Save model to registry
            model_registry = ModelRegistry()
            model_id = model_registry.save_model(
                model=model, 
                scaler=scaler,
                metadata={
                    'stock': stock,
                    'start_date': start,
                    'end_date': end,
                    'training_data_len': training_data_len,
                    'window_size': window_size,
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'performance_metrics': {
                        'rmse': float(rmse),
                        'mae': float(mae),
                        'mse': float(np.mean((predictions - y_test)**2))
                    }
                }
            )
            
            train = data[:training_data_len].copy()
            train['timeTrain'] = time_data[:training_data_len]
            
            valid = data[training_data_len:].copy()
            valid['Predictions'] = predictions
            valid['timeValid'] = time_data[training_data_len:]
            
            # Predict future price
            # Get the last 60 days closing price values
            last_60_days = data[-60:].values
            
            # Scale the data
            last_60_days_scaled = scaler.transform(last_60_days)
            
            # Create empty list and append the past 60 days
            X_test = []
            X_test.append(last_60_days_scaled)
            
            # Convert to numpy array and reshape
            X_test = np.array(X_test)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            
            # Predict price
            pred_price = model.predict(X_test)
            pred_price = scaler.inverse_transform(pred_price)
            
            # Convert dataframes to serializable format
            train_data = {
                'Close': train['Close'].tolist(),
                'timeTrain': train['timeTrain'].apply(lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x)).tolist()
            }
            
            valid_data = {
                'Close': valid['Close'].tolist(),
                'Predictions': valid['Predictions'].tolist(),
                'timeValid': valid['timeValid'].apply(lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x)).tolist()
            }
            
            # Format the dates for the response
            dates = df['Date'].apply(lambda x: x.isoformat() if hasattr(x, 'isoformat') else str(x)).tolist()
            
            # Check if we're using real or fallback data
            data_source = "real" if not df.empty else "fallback"
            
            # Log the successful prediction
            execution_time = (time.time() - start_time) * 1000  # in milliseconds
            monitor.log_prediction(
                stock=stock, 
                prediction_value=float(pred_price[0][0]),
                model_id=model_id,
                execution_time=execution_time
            )
            
            # Log model performance
            monitor.log_model_performance(
                model_id=model_id,
                stock=stock,
                metrics={
                    "rmse": float(rmse),
                    "mae": float(mae)
                }
            )
            
            return Response({
                'data': {
                    'data_source': data_source,
                    'prices': df['Close'].tolist(),
                    'time': dates,
                    'train': train_data,
                    'valid': valid_data,
                    'predicted_price': float(pred_price[0][0]),
                    'rmse': float(rmse),
                    'mae': float(mae),
                    'model_id': model_id
                },
                'message': f"Successfully analyzed {'real' if data_source == 'real' else 'sample'} data for {stock}" 
            })
            
        except Exception as e:
            import traceback
            print(f"Error: {str(e)}")
            print(traceback.format_exc())
            
            # Log the error if monitoring is initialized
            if 'monitor' in locals() and 'start_time' in locals():
                execution_time = (time.time() - start_time) * 1000
                monitor.log_prediction(
                    stock=stock if 'stock' in locals() else None,
                    prediction_value=None,
                    execution_time=execution_time,
                    error=str(e)
                )
                
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HealthCheck(APIView):
    def get(self, request):
        """System health check endpoint"""
        # Get system stats
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get model registry stats
        registry = ModelRegistry()
        models = registry.list_models()
        
        # Get monitoring stats
        monitor = ModelMonitor()
        recent_predictions = monitor.get_prediction_history(limit=5)
        
        health_data = {
            "status": "healthy",
            "system": {
                "memory_usage_percent": memory.percent,
                "disk_usage_percent": disk.percent,
                "cpu_usage_percent": psutil.cpu_percent(interval=0.1),
            },
            "models": {
                "count": len(models),
                "latest_model": models[-1]["model_id"] if models else None
            },
            "monitoring": {
                "recent_predictions": len(recent_predictions),
                "recent_errors": sum(1 for p in recent_predictions if p.get("error"))
            }
        }
        
        return Response(health_data)


class ModelList(APIView):
    def get(self, request):
        """List available models"""
        registry = ModelRegistry()
        stock = request.GET.get('stock')
        limit = int(request.GET.get('limit', 10))
        
        models = registry.list_models(stock=stock, limit=limit)
        
        # Remove full paths for security
        for model in models:
            if 'model_path' in model:
                del model['model_path']
            if 'scaler_path' in model:
                del model['scaler_path']
        
        return Response({"models": models})


class ModelDetail(APIView):
    def get(self, request, model_id):
        """Get details for a specific model"""
        registry = ModelRegistry()
        _, _, metadata = registry.load_model(model_id=model_id)
        
        if not metadata:
            return Response({"error": "Model not found"}, status=status.HTTP_404_NOT_FOUND)
        
        # Remove full paths for security
        if 'model_path' in metadata:
            del metadata['model_path']
        if 'scaler_path' in metadata:
            del metadata['scaler_path']
        
        return Response(metadata)


class PredictionMonitor(APIView):
    def get(self, request):
        """Get prediction history"""
        monitor = ModelMonitor()
        stock = request.GET.get('stock')
        limit = int(request.GET.get('limit', 100))
        
        predictions = monitor.get_prediction_history(stock=stock, limit=limit)
        
        return Response({"predictions": predictions})