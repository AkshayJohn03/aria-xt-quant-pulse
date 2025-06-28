import torch
import joblib
import numpy as np
from src.models.lstm_model import AriaXaTModel

def load_inference_artifacts():
    metadata_path = r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/chunk_metadata_xaat.pkl'
    scaler_path = r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/scaler_xaat_minmaxscaler.pkl'
    model_path = r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/best_aria_xat_model.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metadata = joblib.load(metadata_path)
    scaler = joblib.load(scaler_path)
    features = [f.lower() for f in metadata['features']]
    model = AriaXaTModel(
        input_features=metadata['num_features'],
        hidden_size=128,
        num_layers=2,
        output_classes=3,
        dropout_rate=0.2
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, scaler, features, metadata['lookback_window'], device

def make_prediction(df_slice, model, scaler, features, lookback_window, device):
    if df_slice[features].isnull().any().any():
        return None, None, None
    X = scaler.transform(df_slice[features].values)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        out_class, out_reg = model(X)
        probs = torch.softmax(out_class, dim=1).cpu().numpy().flatten()
        pred_class = probs.argmax()
        confidence = probs.max()
        reg_pred = out_reg.cpu().item()
    return pred_class, confidence, reg_pred 