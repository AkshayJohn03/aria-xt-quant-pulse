import os
import sys
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import optuna
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Import model
from src.models.lstm_model import AriaXaTModel

# --- Load config and metadata ---
config = {
    'processed_data_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/processed_data_npy/All_Indices_XaT_enriched_ohlcv_indicators.joblib',
    'metadata_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/chunk_metadata_xaat.pkl',
    'scaler_path': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models/scaler_xaat_minmaxscaler.pkl',
    'model_save_dir': r'D:/aria/aria-xt-quant-pulse/aria/data/converted/models',
    'batch_size': 256,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

metadata = joblib.load(config['metadata_path'])
scaler = joblib.load(config['scaler_path'])
features = [f.lower() for f in metadata['features']]
lookback_window = metadata['lookback_window']
num_features = metadata['num_features']

# --- Data loading utility ---
def load_sequences(df, features, lookback, scaler):
    arr = df[features].values
    num_seq = arr.shape[0] - lookback
    if num_seq <= 0:
        return None, None, None
    X = np.lib.stride_tricks.sliding_window_view(arr, (lookback, arr.shape[1]))
    X = X.squeeze(1)
    valid = ~np.isnan(X).any(axis=(1,2))
    X = X[valid]
    X_reshaped = X.reshape(-1, X.shape[-1])
    X_scaled = scaler.transform(X_reshaped).reshape(X.shape)
    y_class = df['label_class'].iloc[lookback:lookback+len(valid)][valid].values
    y_reg = df['label_regression_normalized'].iloc[lookback:lookback+len(valid)][valid].values
    return X_scaled, y_class, y_reg

# --- Optuna objective ---
def objective(trial):
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
    dropout = trial.suggest_float('dropout_rate', 0.1, 0.5)
    class_weight = trial.suggest_float('classification_weight', 0.5, 0.9)
    reg_weight = 1.0 - class_weight

    model = AriaXaTModel(
        input_features=num_features,
        hidden_size=128,
        num_layers=2,
        output_classes=3,
        dropout_rate=dropout
    ).to(config['device'])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    # Load data
    df = joblib.load(config['processed_data_path'])
    df.columns = [c.lower() for c in df.columns]
    train_size = int(0.8 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    X_train, y_class_train, y_reg_train = load_sequences(train_df, features, lookback_window, scaler)
    X_val, y_class_val, y_reg_val = load_sequences(val_df, features, lookback_window, scaler)
    if X_train is None or X_val is None:
        return float('inf')
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_class_train, dtype=torch.long), torch.tensor(y_reg_train, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_class_val, dtype=torch.long), torch.tensor(y_reg_val, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], shuffle=False)

    best_val_loss = float('inf')
    for epoch in range(8):
        model.train()
        for xb, yb_class, yb_reg in train_loader:
            xb, yb_class, yb_reg = xb.to(config['device']), yb_class.to(config['device']), yb_reg.to(config['device'])
            optimizer.zero_grad()
            out_class, out_reg = model(xb)
            loss_class = criterion_class(out_class, yb_class)
            loss_reg = criterion_reg(out_reg, yb_reg)
            loss = class_weight * loss_class + reg_weight * loss_reg
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        val_loss = 0
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb_class, yb_reg in val_loader:
                xb, yb_class, yb_reg = xb.to(config['device']), yb_class.to(config['device']), yb_reg.to(config['device'])
                out_class, out_reg = model(xb)
                loss_class = criterion_class(out_class, yb_class)
                loss_reg = criterion_reg(out_reg, yb_reg)
                loss = class_weight * loss_class + reg_weight * loss_reg
                val_loss += loss.item()
                preds.extend(out_class.argmax(dim=1).cpu().numpy())
                trues.extend(yb_class.cpu().numpy())
        val_loss /= len(val_loader)
        f1 = f1_score(trues, preds, average='weighted', zero_division=0)
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    return best_val_loss

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    print('Best trial:', study.best_trial.value)
    print('Params:', study.best_trial.params) 