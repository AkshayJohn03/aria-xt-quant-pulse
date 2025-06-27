from flask import Flask, request, jsonify, render_template
import pandas as pd
from datetime import datetime, timedelta
import logging
import argparse
from typing import Dict, Any
import os
from dotenv import load_dotenv
import asyncio

from utils.data_loader import DataLoader
from utils.backtester import Backtester
from models.strategy_engine import StrategyEngine, StrategyConfig

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize components
data_loader = DataLoader(api_key=os.getenv('TWELVEDATA_API_KEY'))
strategy_engine = StrategyEngine()

@app.route('/')
def index():
    """Render the main dashboard."""
    return render_template('index.html')

@app.route('/backtest', methods=['POST'])
async def run_backtest():
    """
    Run backtesting simulation with provided parameters.
    
    Expected JSON payload:
    {
        "symbol": "NIFTY",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "strategy_profile": "hybrid",
        "mode": "safe",
        "initial_capital": 100000,
        "max_trades_per_day": 3,
        "risk_per_trade": 0.02
    }
    """
    try:
        # Get parameters from request
        params = request.get_json()
        
        # Validate required parameters
        required_params = ['symbol', 'start_date', 'end_date']
        if not all(param in params for param in required_params):
            return jsonify({
                'error': f'Missing required parameters. Required: {required_params}'
            }), 400
        
        # Parse dates
        start_date = datetime.strptime(params['start_date'], '%Y-%m-%d')
        end_date = datetime.strptime(params['end_date'], '%Y-%m-%d')
        
        # Fetch historical data
        historical_data = await data_loader.load_ohlc_from_twelvedata(
            symbol=params['symbol'],
            interval='1min',
            start_date=params['start_date'],
            end_date=params['end_date']
        )
        
        # Initialize backtester with parameters
        backtester = Backtester(
            historical_ohlc_data=historical_data,
            initial_capital=params.get('initial_capital', 100000),
            max_trades_per_day=params.get('max_trades_per_day', 3),
            risk_per_trade=params.get('risk_per_trade', 0.02)
        )
        
        # Generate signals using strategy engine
        signals = strategy_engine.generate_signals(
            df=historical_data,
            strategy_profile=params.get('strategy_profile', 'hybrid'),
            mode=params.get('mode', 'safe')
        )
        
        # Run backtest simulation
        backtester.simulate_trades(signals)
        
        # Get results
        pnl_log = backtester.generate_pnl_log()
        metrics = backtester.calculate_metrics()
        
        # Prepare response
        response = {
            'metrics': {
                'total_trades': metrics['total_trades'],
                'win_rate': round(metrics['win_rate'], 2),
                'total_pnl': round(metrics['total_pnl'], 2),
                'max_drawdown': round(metrics['max_drawdown'], 2),
                'sharpe_ratio': round(metrics['sharpe_ratio'], 2),
                'avg_profit_per_trade': round(metrics['avg_profit_per_trade'], 2),
                'avg_trade_duration': str(metrics['avg_trade_duration'])
            },
            'trades': pnl_log.to_dict(orient='records')
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in backtesting: {str(e)}")
        return jsonify({'error': str(e)}), 500

async def run_cli_backtest(args: argparse.Namespace) -> None:
    """Run backtesting from command line."""
    try:
        # Fetch historical data
        historical_data = await data_loader.load_ohlc_from_twelvedata(
            symbol=args.symbol,
            interval='1min',
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Initialize backtester
        backtester = Backtester(
            historical_ohlc_data=historical_data,
            initial_capital=args.initial_capital,
            max_trades_per_day=args.max_trades_per_day,
            risk_per_trade=args.risk_per_trade
        )
        
        # Generate signals
        signals = strategy_engine.generate_signals(
            df=historical_data,
            strategy_profile=args.strategy_profile,
            mode=args.mode
        )
        
        # Run simulation
        backtester.simulate_trades(signals)
        
        # Print results
        metrics = backtester.calculate_metrics()
        print("\nBacktesting Results:")
        print("===================")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2f}%")
        print(f"Total P&L: ₹{metrics['total_pnl']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Average Profit per Trade: ₹{metrics['avg_profit_per_trade']:.2f}")
        print(f"Average Trade Duration: {metrics['avg_trade_duration']}")
        
    except Exception as e:
        logger.error(f"Error in CLI backtesting: {str(e)}")
        raise

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Aria XT Quant Pulse Trading Bot')
    
    # Add backtesting arguments
    parser.add_argument('--backtest', action='store_true', help='Run in backtest mode')
    parser.add_argument('--symbol', type=str, help='Trading symbol (e.g., NIFTY)')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument(
        '--strategy-profile',
        type=str,
        choices=['ai-only', 'fallback-only', 'hybrid'],
        default='hybrid',
        help='Strategy profile to use'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['safe', 'aggressive'],
        default='safe',
        help='Trading mode'
    )
    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='Initial capital for backtesting'
    )
    parser.add_argument(
        '--max-trades-per-day',
        type=int,
        default=3,
        help='Maximum trades per day'
    )
    parser.add_argument(
        '--risk-per-trade',
        type=float,
        default=0.02,
        help='Risk per trade as fraction of capital'
    )
    
    args = parser.parse_args()
    
    if args.backtest:
        # Validate backtesting arguments
        if not all([args.symbol, args.start_date, args.end_date]):
            parser.error("--symbol, --start-date, and --end-date are required for backtesting")
        asyncio.run(run_cli_backtest(args))
    else:
        # Run Flask app
        app.run(debug=True) 