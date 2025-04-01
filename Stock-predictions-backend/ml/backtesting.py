# Stock-predictions-backend/ml/backtesting.py
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime, timedelta
from .model_registry import ModelRegistry
import tensorflow as tf
import math
from sklearn.preprocessing import MinMaxScaler

class ModelBacktester:
    def __init__(self, output_dir="backtest_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.registry = ModelRegistry()
        
    def backtest_specific_model(self, model_id, test_period_days=30, use_sample_data=False):
        """Backtest a specific model from the registry"""
        model, scaler, metadata = self.registry.load_model(model_id=model_id)
        
        if not model or not metadata:
            print(f"Model {model_id} not found")
            return None
            
        stock = metadata.get('stock')
        if not stock:
            print(f"No stock symbol found for model {model_id}")
            return None
            
        return self.run_backtest(model, scaler, stock, test_period_days, model_id, use_sample_data)
        
    def backtest_stock(self, stock, test_period_days=30, use_sample_data=False):
        """Backtest the latest model for a stock"""
        model, scaler, metadata = self.registry.load_model(stock=stock)
        
        if not model:
            print(f"No model found for stock {stock}")
            return None
            
        model_id = metadata.get('model_id')
        return self.run_backtest(model, scaler, stock, test_period_days, model_id, use_sample_data)
    
    def run_backtest(self, model, scaler, stock, test_period_days, model_id, use_sample_data=False):
        """Run the actual backtest"""
        # Set up date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=test_period_days + 60)  # Need 60 days of history for each prediction
        
        # Try to get real data if not using sample data
        df = pd.DataFrame()
        if not use_sample_data:
            try:
                print(f"Downloading data for {stock} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
                df = yf.download(stock, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
                
                if df.empty:
                    print(f"No data available for {stock}, falling back to sample data")
                    use_sample_data = True
            except Exception as e:
                print(f"Error downloading data: {str(e)}, falling back to sample data")
                use_sample_data = True
        
        # Generate sample data if needed
        if use_sample_data:
            print(f"Using synthetic data for backtesting {stock}")
            date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
            
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
            
            print(f"Created synthetic data with {len(df)} records")
        
        # Prepare for predictions
        predictions = []
        actuals = []
        dates = []
        
        # We'll use a sliding window approach - test the model on each day
        # in the test period using data up to the previous day
        window_size = 60  # Default window size if not in metadata
        
        # Ensure we have enough days
        if len(df) < window_size + 10:
            print(f"Not enough data points for backtesting (need at least {window_size+10}, got {len(df)})")
            return None
        
        # Split the data
        test_start_idx = max(window_size, len(df) - test_period_days)
        
        for i in range(test_start_idx, len(df)):
            test_date = df.index[i]
            # Get data up to the day before test_date
            test_data = df.iloc[:i].copy()
            
            if len(test_data) < window_size:  # Need at least window_size days of data
                continue
                
            # Get the actual price on test_date
            actual_price = df.iloc[i]['Close']
            
            # Prepare input for the model (last window_size days)
            close_data = test_data['Close'].values[-window_size:].reshape(-1, 1)
            scaled_data = scaler.transform(close_data)
            
            # Create the input tensor
            X_test = np.array([scaled_data])
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            
            # Make prediction
            pred_scaled = model.predict(X_test, verbose=0)
            pred_price = scaler.inverse_transform(pred_scaled)[0][0]
            
            # Store results
            predictions.append(pred_price)
            actuals.append(actual_price)
            dates.append(test_date)
        
        # Ensure we have at least some predictions
        if len(predictions) == 0:
            print("No predictions were generated during backtesting")
            return None
            
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # Calculate MAPE, handling possible zeros in actuals
        with np.errstate(divide='ignore', invalid='ignore'):
            mape = np.mean(np.abs((np.array(actuals) - np.array(predictions)) / np.array(actuals))) * 100
            if np.isnan(mape) or np.isinf(mape):
                mape = 0.0
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'Date': dates,
            'Actual': actuals,
            'Predicted': predictions,
            'Error': np.array(actuals) - np.array(predictions),
            'Percent_Error': (np.array(actuals) - np.array(predictions)) / np.array(actuals) * 100
        })
        
        # Replace infinities and NaNs in Percent_Error with 0
        results_df['Percent_Error'].replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f"{self.output_dir}/{stock}_{model_id}_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)
        
        # Generate plot
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(dates, actuals, 'b-', label='Actual')
        plt.plot(dates, predictions, 'r--', label='Predicted')
        plt.title(f"{stock} - Actual vs Predicted Prices (Model: {model_id})")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(dates, results_df['Percent_Error'], 'g-')
        plt.axhline(y=0, color='k', linestyle='-')
        plt.title('Prediction Error (%)')
        plt.grid(True)
        
        plot_path = f"{self.output_dir}/{stock}_{model_id}_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        
        # Compile report
        report = {
            "stock": stock,
            "model_id": model_id,
            "test_period_days": test_period_days,
            "metrics": {
                "rmse": rmse,
                "mae": mae,
                "mape": mape,
                "r2": r2
            },
            "results_file": results_path,
            "plot_file": plot_path,
            "timestamp": timestamp,
            "used_sample_data": use_sample_data
        }
        
        # Save report
        report_path = f"{self.output_dir}/{stock}_{model_id}_{timestamp}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Backtest completed for {stock} using model {model_id}")
        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, MAPE: {mape:.2f}%, R²: {r2:.4f}")
        print(f"Results saved to {results_path}")
        print(f"Plot saved to {plot_path}")
        
        return report