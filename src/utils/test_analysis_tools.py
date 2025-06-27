#!/usr/bin/env python3
"""
Test script for Aria-XsT Analysis Tools
Demonstrates the functionality of the analysis tools module.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.analysis_tools import fetch_data_with_fallback, verify_aggressive_labeling, simulate_trading_day

def test_data_fetching():
    """Test the data fetching functionality with fallback support."""
    print("=" * 60)
    print("Testing Data Fetching with Fallback Support")
    print("=" * 60)
    
    # Test symbols that should work with Twelvedata
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in test_symbols:
        print(f"\nTesting {symbol}...")
        try:
            df = fetch_data_with_fallback(symbol, '1min')
            if df is not None:
                print(f"✅ Successfully fetched {len(df)} rows for {symbol}")
                print(f"   Latest close: ${df['Close'].iloc[-1]:.2f}")
            else:
                print(f"❌ Failed to fetch data for {symbol}")
        except Exception as e:
            print(f"❌ Error fetching {symbol}: {e}")
    
    # Test NIFTY (should fallback to Yahoo Finance)
    print(f"\nTesting NIFTY (should use Yahoo Finance fallback)...")
    try:
        df = fetch_data_with_fallback('NIFTY', '1min')
        if df is not None:
            print(f"✅ Successfully fetched {len(df)} rows for NIFTY via Yahoo Finance")
            print(f"   Latest close: ₹{df['Close'].iloc[-1]:.2f}")
        else:
            print(f"❌ Failed to fetch data for NIFTY")
    except Exception as e:
        print(f"❌ Error fetching NIFTY: {e}")

def test_verify_labeling():
    """Test the labeling verification function."""
    print("\n" + "=" * 60)
    print("Testing Labeling Verification")
    print("=" * 60)
    
    # This would require actual processed data files
    print("Note: This function requires actual processed .joblib files.")
    print("To test this function, you would need to:")
    print("1. Have processed data files in the backend/processed_data_npy directory")
    print("2. Call: verify_aggressive_labeling(sample_file_name='your_file.joblib')")
    print("Example:")
    print("verify_aggressive_labeling(sample_file_name='NIFTY_50_1min_processed.joblib', num_rows=5)")

def test_simulation():
    """Test the trading simulation function."""
    print("\n" + "=" * 60)
    print("Testing Trading Simulation")
    print("=" * 60)
    
    print("Note: This function runs a live simulation for the specified duration.")
    print("To test this function, you would call:")
    print("simulate_trading_day(symbol='AAPL', interval='1min', duration_minutes=2)")
    print("\nThis will:")
    print("- Load the trained model")
    print("- Fetch live data every minute")
    print("- Make predictions")
    print("- Display results in real-time")
    print("\n⚠️  Warning: This uses real API calls and may consume API quota.")

if __name__ == "__main__":
    print("Aria-XsT Analysis Tools Test Suite")
    print("=" * 60)
    
    # Test data fetching
    test_data_fetching()
    
    # Test labeling verification (info only)
    test_verify_labeling()
    
    # Test simulation (info only)
    test_simulation()
    
    print("\n" + "=" * 60)
    print("Test Suite Complete!")
    print("=" * 60)
    print("\nTo run actual simulations:")
    print("1. For data fetching test: python -c \"from utils.analysis_tools import fetch_data_with_fallback; df = fetch_data_with_fallback('AAPL', '1min'); print(f'Fetched {len(df)} rows')\"")
    print("2. For simulation: python -c \"from utils.analysis_tools import simulate_trading_day; simulate_trading_day(symbol='AAPL', interval='1min', duration_minutes=2)\"") 