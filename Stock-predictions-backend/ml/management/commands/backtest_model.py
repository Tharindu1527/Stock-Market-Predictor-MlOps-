# Stock-predictions-backend/ml/management/commands/backtest_model.py
from django.core.management.base import BaseCommand, CommandError
from ml.backtesting import ModelBacktester
import json

class Command(BaseCommand):
    help = 'Backtest a stock prediction model'

    def add_arguments(self, parser):
        parser.add_argument('--stock', type=str, help='Stock symbol to backtest')
        parser.add_argument('--model-id', type=str, help='Specific model ID to backtest')
        parser.add_argument('--days', type=int, default=30, help='Number of days to backtest')
        parser.add_argument('--output-format', choices=['text', 'json'], default='text', 
                            help='Output format (text or json)')
        parser.add_argument('--use-sample-data', action='store_true', 
                            help='Use synthetic data instead of real stock data')

    def handle(self, *args, **options):
        backtester = ModelBacktester()
        
        stock = options.get('stock')
        model_id = options.get('model_id')
        days = options.get('days')
        output_format = options.get('output_format')
        use_sample_data = options.get('use_sample_data')
        
        if not stock and not model_id:
            raise CommandError("Either --stock or --model-id must be specified")
        
        if model_id:
            report = backtester.backtest_specific_model(model_id, days, use_sample_data)
        else:
            report = backtester.backtest_stock(stock, days, use_sample_data)
        
        if not report:
            self.stderr.write(self.style.ERROR('Backtesting failed, see errors above.'))
            return
        
        if output_format == 'json':
            self.stdout.write(json.dumps(report, indent=2))
        else:
            self.stdout.write(self.style.SUCCESS(f"Backtest completed for {report['stock']}"))
            self.stdout.write(f"Model: {report['model_id']}")
            self.stdout.write(f"Period: {report['test_period_days']} days")
            self.stdout.write("\nMetrics:")
            self.stdout.write(f"  RMSE: {report['metrics']['rmse']:.4f}")
            self.stdout.write(f"  MAE: {report['metrics']['mae']:.4f}")
            self.stdout.write(f"  MAPE: {report['metrics']['mape']:.2f}%")
            self.stdout.write(f"  R²: {report['metrics']['r2']:.4f}")
            self.stdout.write("\nResults:")
            self.stdout.write(f"  CSV: {report['results_file']}")
            self.stdout.write(f"  Plot: {report['plot_file']}")