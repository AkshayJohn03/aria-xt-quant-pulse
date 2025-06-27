import os
import sys
import pandas as pd
import numpy as np
import torch
import joblib
import yfinance as yf
import ta
from twelvedata import TDClient
from sklearn.preprocessing import MinMaxScaler
import warnings

# Ensure root directory is in path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the LSTMModel from its dedicated file
from src.models.lstm_model import AriaXaTModel 

# Suppress sklearn UserWarning about feature names
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- Configuration (Restored) ---
config = {
    'local_models_save_dir': r'D:\aria\aria-xt-quant-pulse\aria\data\converted\models',
    'local_processed_data_dir': r'D:\aria\aria-xt-quant-pulse\aria\data\converted\processed_data_npy',
    'features': [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'EMA_Fast', 'EMA_Slow', 'MACD_Signal', 'RSI', 'ADX', 'ATR',
        'Supertrend', 'Supertrend_Dir_Binary'
    ],
    'label_to_use': 'label_aggressive',
    'lookback_window': 60,
    'hidden_dim': 128,
    'num_layers': 2,
    'output_dim': 2,
    'twelvedata_api_key': os.getenv('TWELVEDATA_API_KEY', 'YOUR_TWELVEDATA_API_KEY'),
    'target_symbol': 'NIFTY/INDEX',
    'target_interval': '1min',
}

# --- Load metadata for model input dimensions and lookback window ---
metadata_path = r'D:\aria\aria-xt-quant-pulse\aria\data\converted\models\chunk_metadata_xaat.pkl'
best_model_path = r'D:\aria\aria-xt-quant-pulse\aria\data\converted\models\best_aria_xat_model.pth'
metadata = joblib.load(metadata_path)
input_features = metadata['num_features']
lookback_window = metadata['lookback_window']
features = metadata['features']

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = AriaXaTModel(
    input_features=input_features,
    hidden_size=128,  # match aria_xat_training.py config
    num_layers=2,
    output_classes=3,
    dropout_rate=0.2
).to(device)

# Load the trained weights
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()

# --- Model Loading ---
def load_inference_artifacts(model_dir, input_dim, hidden_dim, num_layers, output_dim, device):
    """
    Loads the trained AriaXaTModel and scaler.
    """
    model_path = os.path.join(model_dir, 'best_aria_xat_model.pth')
    scaler_path = os.path.join(model_dir, 'scaler_xaat_minmaxscaler.pkl')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at: {scaler_path}")

    model = AriaXaTModel(
        input_features=input_dim,
        hidden_size=hidden_dim,
        num_layers=num_layers,
        output_classes=3,
        dropout_rate=0.2
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Model loaded from: {model_path}")

    scaler = joblib.load(scaler_path)
    print(f"Scaler loaded from: {scaler_path}")

    return model, scaler

# --- Data Fetching and Preprocessing ---
def fetch_and_preprocess_data(symbol: str = None, interval: str = None, lookback_window: int = 60,
                              features_list: list = None, scaler=None, td_api_key: str = None,
                              df_input: pd.DataFrame = None):
    df = df_input.copy() if df_input is not None else None

    if df is None:
        print(f"Attempting to fetch data for {symbol} at {interval}...")
        try:
            td = TDClient(apikey=td_api_key)
            fetch_count = max(lookback_window + 100, 250)
            ts = td.time_series(symbol=symbol, interval=interval, outputsize=fetch_count, timezone="exchange")
            df = ts.as_pandas()
            df = df.iloc[::-1].copy() # Reverse to chronological
            print(f"Fetched {len(df)} bars from Twelvedata for {symbol}.")
        except Exception as e:
            print(f"Twelvedata fetch failed: {e}. Falling back to yfinance.")
            try:
                yf_interval_map = {'1min': '1m', '5min': '5m', '15min': '15m', '1h': '1h', '1d': '1d'}
                yf_period = "5d" if interval in ['1min', '5min', '15min'] else "1y"
                df = yf.download(symbol, period=yf_period, interval=yf_interval_map.get(interval, interval))
                print(f"Fetched {len(df)} bars from Yahoo Finance for {symbol}.")
            except Exception as e_yf:
                print(f"Error fetching data from Yahoo Finance for {symbol}: {e_yf}")
                return None, None
    
    if df is None or df.empty:
        print("DataFrame is empty after fetching.")
        return None, None

    df.rename(columns={c.lower(): c.capitalize() for c in df.columns}, inplace=True)

    # Debug: Show NaN counts before any cleaning
    print("NaN counts per feature before cleaning:")
    print(df[features_list].isna().sum())
    print(f"Rows before cleaning: {len(df)}")

    all_features_present = all(f in df.columns for f in features_list)

    if not all_features_present:
        print("Calculating technical indicators...")
        df['EMA_Fast'] = ta.ema(df['Close'], length=12)
        df['EMA_Slow'] = ta.ema(df['Close'], length=26)
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd is not None and 'MACDs_12_26_9' in macd.columns:
            df['MACD_Signal'] = macd['MACDs_12_26_9']
        else:
            df['MACD_Signal'] = np.nan
        df['RSI'] = ta.rsi(df['Close'], length=14)
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx is not None and 'ADX_14' in adx.columns:
            df['ADX'] = adx['ADX_14']
        else:
            df['ADX'] = np.nan
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        st = ta.supertrend(df['High'], df['Low'], df['Close'], length=10, multiplier=3)
        if st is not None and 'SUPERT_10_3.0' in st.columns:
            df['Supertrend'] = st['SUPERT_10_3.0']
            df['Supertrend_Dir_Binary'] = np.where(df['Close'] > df['Supertrend'], 1, -1)
        else:
            df['Supertrend'] = np.nan
            df['Supertrend_Dir_Binary'] = np.nan
    else:
        print("All required features already present. Skipping indicator calculation.")

    # Debug: Show NaN counts after indicator calculation
    print("NaN counts per feature after indicator calculation:")
    print(df[features_list].isna().sum())
    print(f"Rows before dropna: {len(df)}")

    df.dropna(inplace=True)

    # Debug: Show rows after dropna
    print(f"Rows after dropna: {len(df)}")

    if len(df) < lookback_window:
        print(f"After cleaning, not enough data ({len(df)}) for lookback window ({lookback_window}).")
        return None, None

    current_close_price = df['Close'].iloc[-1]
    
    available_features = [f for f in features_list if f in df.columns]
    if len(available_features) != len(features_list):
        missing = set(features_list) - set(available_features)
        print(f"Warning: Features missing from DataFrame: {missing}. This might cause scaling issues.")
        return None, None

    data_for_scaling = df[features_list]
    
    try:
        scaled_features = scaler.transform(data_for_scaling)
    except Exception as e:
        print(f"Error during scaling: {e}")
        return None, None

    if len(scaled_features) < lookback_window:
        return None, None

    input_sequence = scaled_features[-lookback_window:]
    input_sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)

    return input_sequence, current_close_price

def make_prediction(model, sequence_tensor):
    if sequence_tensor is None or sequence_tensor.shape[0] == 0:
        return None, None
    with torch.no_grad():
        outputs = model(sequence_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    return predicted_class, confidence 