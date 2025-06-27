import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
# Load .env from backend directory
load_dotenv(dotenv_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '../../backend/.env')))

# Import the utility functions and configuration from the new utils file
from inference.inference_utils import load_inference_artifacts, fetch_and_preprocess_data, make_prediction, config, device
from models.lstm_model import AriaXaTModel
import torch
import joblib

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

if __name__ == "__main__":
    print("-" * 75)
    print(" Aria-XsT Inference Pipeline")
    print("-" * 75)
    print(" 1. Ensure your Twelvedata API Key is set in the config.")
    print(" 2. The script expects the trained model (best_aria_xat_model.pth) and scaler (scaler_xaat_minmaxscaler.pkl) to be in:")
    print(f"    {config['local_models_save_dir']}")
    print(" 3. Adjust 'target_symbol' and 'target_interval' in the config.")
    print("-" * 75)
    input_sequence = fetch_and_preprocess_data(
        config['target_symbol'],
        config['target_interval'],
        config['lookback_window'],
        config['features'],
        scaler,
        config['twelvedata_api_key']
    )
    
    # Handle the new return format (sequence_tensor, current_close_price)
    if isinstance(input_sequence, tuple):
        sequence_tensor, current_close_price = input_sequence
    else:
        sequence_tensor = input_sequence
        current_close_price = None
    
    # Make prediction (classification and regression)
    with torch.no_grad():
        class_logits, reg_output = model(sequence_tensor.unsqueeze(0).to(device))
        class_probs = torch.softmax(class_logits, dim=1)
        predicted_class = class_probs.argmax(dim=1).item()
        confidence = class_probs.max(dim=1).values.item()
        predicted_movement = reg_output.item()
    
    print(f"\nModel Prediction for {config['target_symbol']} ({config['target_interval']}):")
    if predicted_class is not None:
        if predicted_class == 2:
            prediction_label = "BUY CALL"
        elif predicted_class == 0:
            prediction_label = "BUY PUT"
        else:
            prediction_label = "HOLD"
        print(f"Predicted Class: {prediction_label} (Confidence: {confidence:.2%})")
        print(f"Predicted Movement (regression): {predicted_movement:.4f}")
    else:
        print("Could not make a prediction.") 