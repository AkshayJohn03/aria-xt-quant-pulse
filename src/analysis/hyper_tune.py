import os
import sys
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from collections import deque
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import optuna

# --- Ensure the project root is in the Python path for imports ---
# Correcting the path to go up two directories from src/analysis to the project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import core model. config and device from inference_utils will be used but
# fetch_and_preprocess_data will be re-defined for clarity and control.
from src.models.lstm_model import AriaXaTModel
from src.inference.inference_utils import config, device

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler('hyper_tune_log.log', mode='a')
                    ])
logger = logging.getLogger('HyperTuneRunner')
logger.setLevel(logging.INFO) # Set to INFO for general runs, DEBUG for verbose debugging

# --- Configuration (using existing config for paths and static params) ---
# Ensure these paths and values are correctly set in your config.py
# For tuning, we'll suggest ranges for some of these.
MODEL_DIR = config['local_models_save_dir']
# FIXED PATH: Use config['local_processed_data_dir'] directly, do NOT append 'processed_data_npy' again
PROCESSED_DATA_PATH = os.path.join(
    config['local_processed_data_dir'],
    'All_Indices_XaT_enriched_ohlcv_indicators.joblib'
)
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler_xaat_minmaxscaler.pkl')
METADATA_PATH = os.path.join(MODEL_DIR, 'chunk_metadata_xaat.pkl')
BEST_MODEL_CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'best_aria_xat_model.pth')

# Static model parameters from your config (these are usually not hyper-tuned unless structural change)
HIDDEN_SIZE = config['hidden_dim']
NUM_LAYERS = config['num_layers']
OUTPUT_CLASSES = 3 # Assuming BUY_PUT, HOLD, BUY_CALL

# --- Re-define fetch_and_preprocess_data for explicit control ---
# This version ensures column casing is handled correctly within the tuning script itself.
def fetch_and_preprocess_data_for_tuning(df_input, lookback_window, features_list, scaler):
    """
    Preprocesses a DataFrame slice for model input.
    Ensures that the input DataFrame has columns matching the features_list (lowercase).
    """
    if df_input.empty:
        logger.warning("Input DataFrame for preprocessing is empty.")
        return None, None

    # Convert all columns to lowercase for consistency with features_list
    df_input.columns = [col.lower() for col in df_input.columns]

    # Ensure all required features are present and handle potential NaNs
    # IMPORTANT: The model's features_list should be used here.
    df_features = df_input[features_list].copy()

    # Drop rows with any NaN values in the feature set *for this specific slice*
    # This prevents creating sequences with missing data.
    initial_rows = len(df_features)
    df_features.dropna(inplace=True)
    if len(df_features) == 0:
        logger.warning(f"After cleaning, not enough data ({len(df_features)}) for lookback window ({lookback_window}). Original rows: {initial_rows}. Skipping prediction.")
        return None, None
    elif len(df_features) < lookback_window:
        logger.warning(f"After cleaning, data ({len(df_features)}) is less than lookback window ({lookback_window}). Skipping prediction.")
        return None, None

    # Get the current close price (assuming 'close' is in features_list and lowercase)
    current_close_price = df_input['close'].iloc[-1] # Use the original 'close' from df_input

    # Prepare sequences for the model
    sequences = []
    # Loop from the first possible sequence start to the last possible sequence end
    # to generate the full set of sequences required for a training/validation batch
    # We need at least `lookback_window` rows to form a sequence.
    for i in range(len(df_features) - lookback_window + 1):
        sequence = df_features.iloc[i : i + lookback_window]
        sequences.append(sequence.values) # Convert to numpy array

    if not sequences:
        logger.warning("No sequences could be formed from the preprocessed data.")
        return None, None
    
    # Scale the sequences. Scaler must be fitted on training data.
    # Reshape for scaler: (num_samples * lookback_window, num_features)
    # Then inverse reshape: (num_samples, lookback_window, num_features)
    scaled_sequences = []
    for seq in sequences:
        scaled_seq = scaler.transform(seq)
        scaled_sequences.append(scaled_seq)

    # Convert list of scaled sequences to a single numpy array then to torch tensor
    input_tensor = torch.tensor(np.array(scaled_sequences), dtype=torch.float32)

    return input_tensor, current_close_price

# --- Load initial artifacts (metadata and scaler) outside objective for efficiency ---
try:
    METADATA = joblib.load(METADATA_PATH)
    SCALER = joblib.load(SCALER_PATH)
    # Ensure scaler is indeed a MinMaxScaler or compatible for .transform
    if not isinstance(SCALER, MinMaxScaler):
        logger.error(f"Loaded scaler is not a MinMaxScaler. Type: {type(SCALER)}. Please check scaler_xaat_minmaxscaler.pkl.")
        sys.exit(1)

    MODEL_INPUT_FEATURES = METADATA['num_features']
    LOOKBACK_WINDOW = METADATA['lookback_window']
    MODEL_FEATURES_LIST = METADATA['features'] # This is the original list, potentially Title Case
    MODEL_FEATURES_LIST = [f.lower() for f in MODEL_FEATURES_LIST] # Ensure it's all lowercase
    logger.info(f"Loaded metadata. Model expects {MODEL_INPUT_FEATURES} features and lookback {LOOKBACK_WINDOW}.")
    logger.debug(f"Model features list from metadata: {MODEL_FEATURES_LIST}")

except Exception as e:
    logger.error(f"Failed to load essential artifacts (metadata or scaler): {e}")
    sys.exit(1)


# --- Objective function for Optuna ---
def objective(trial):
    logger.info(f"Starting Optuna trial {trial.number}...")

    # Define hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True) # Fine-tuning LR
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5, step=0.05)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True) # L2 regularization
    classification_weight = trial.suggest_float("classification_weight", 0.5, 0.9, step=0.1)
    regression_weight = 1.0 - classification_weight # Ensure weights sum to 1

    # Load the best pre-trained model for a warm start
    model = AriaXaTModel(
        input_features=MODEL_INPUT_FEATURES,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_classes=OUTPUT_CLASSES,
        dropout_rate=dropout_rate # Use suggested dropout
    ).to(device)

    try:
        model.load_state_dict(torch.load(BEST_MODEL_CHECKPOINT_PATH, map_location=device))
        logger.info(f"Trial {trial.number}: Warm-started model from {BEST_MODEL_CHECKPOINT_PATH}")
    except Exception as e:
        logger.error(f"Trial {trial.number}: Could not load best model checkpoint for warm start: {e}. Initializing from scratch.")
        # If loading fails, the model remains with random initialization, which is less efficient.
        # Ensure 'best_aria_xat_model.pth' exists and is compatible.

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Loss functions
    class_criterion = nn.CrossEntropyLoss()
    reg_criterion = nn.MSELoss()

    # --- Data Loading and Splitting (within objective for self-containment) ---
    try:
        full_df = joblib.load(PROCESSED_DATA_PATH)
        # Standardize all columns to lowercase upon loading
        full_df.columns = [col.lower() for col in full_df.columns]
        logger.debug(f"Trial {trial.number}: Columns of full_df after lowercasing: {full_df.columns.tolist()}")

        # Ensure DatetimeIndex, crucial for time-based splitting
        if not isinstance(full_df.index, pd.DatetimeIndex):
            # FIX: Create naive datetime first, then localize it
            try:
                naive_start_dt = pd.to_datetime('2024-01-01 09:15:00') 
                start_dt_localized = naive_start_dt.tz_localize('Asia/Kolkata')
                full_df.index = pd.date_range(start=start_dt_localized, periods=len(full_df), freq='1min') 
                logger.warning(f"Trial {trial.number}: Generated a dummy DatetimeIndex for data. For production, ensure real timestamps.")
            except Exception as e:
                logger.error(f"Trial {trial.number}: Could not create DatetimeIndex: {e}")
                return float('inf') # Return large loss to discard trial

        # Filter for NIFTY Market Trading Hours (9:15 AM to 3:30 PM IST)
        market_hours_df = full_df.between_time('09:15', '15:30')
        logger.info(f"Trial {trial.number}: Filtered data to {len(market_hours_df)} bars within NIFTY trading hours.")

        if len(market_hours_df) < LOOKBACK_WINDOW + (2 * 26) + 1000: # Ensure enough data for splits and lookback
            logger.error(f"Trial {trial.number}: Not enough data after filtering for training/validation. Available: {len(market_hours_df)}")
            return float('inf')

        # Time-based splitting for train and validation (no test set here)
        # Use first 80% for training, next 20% for validation (chronological split)
        train_size = int(len(market_hours_df) * 0.8)
        train_df_raw = market_hours_df.iloc[:train_size].copy()
        val_df_raw = market_hours_df.iloc[train_size:].copy()

        # Optionally, for quick testing, uncomment the following lines to use a smaller subset:
        # train_df_raw = train_df_raw.iloc[:100000]
        # val_df_raw = val_df_raw.iloc[:20000]
        logger.info(f"Trial {trial.number}: Train data size: {len(train_df_raw)}, Validation data size: {len(val_df_raw)}")

        # --- Vectorized Sequence Generation for Train/Validation ---
        def create_sequences_vectorized(df, features, lookback_window, scaler):
            arr = df[features].values
            num_sequences = arr.shape[0] - lookback_window
            if num_sequences <= 0:
                return [], [], []
            # Create rolling windows
            shape = (num_sequences, lookback_window, arr.shape[1])
            strides = (arr.strides[0], arr.strides[0], arr.strides[1])
            X = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
            # Find valid (no NaN) sequences
            valid_mask = ~np.isnan(X).any(axis=(1,2))
            X = X[valid_mask]
            # Scale all valid sequences in batch
            if len(X) == 0:
                return [], [], []
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = scaler.transform(X_reshaped).reshape(X.shape)
            # Get targets
            y_class = df['label_class'].iloc[lookback_window:lookback_window+len(valid_mask)][valid_mask].values
            y_reg = df['label_regression_normalized'].iloc[lookback_window:lookback_window+len(valid_mask)][valid_mask].values
            return X_scaled, y_class, y_reg

        logger.info(f"Trial {trial.number}: Generating train sequences (vectorized)...")
        X_train, y_class_train, y_reg_train = create_sequences_vectorized(train_df_raw, MODEL_FEATURES_LIST, LOOKBACK_WINDOW, SCALER)
        logger.info(f"Trial {trial.number}: Generated {len(X_train)} train sequences.")
        logger.info(f"Trial {trial.number}: Generating validation sequences (vectorized)...")
        X_val, y_class_val, y_reg_val = create_sequences_vectorized(val_df_raw, MODEL_FEATURES_LIST, LOOKBACK_WINDOW, SCALER)
        logger.info(f"Trial {trial.number}: Generated {len(X_val)} validation sequences.")

        if len(X_train) == 0 or len(X_val) == 0:
            logger.error(f"Trial {trial.number}: No valid sequences generated for train or validation after NaN handling and slicing. Train sequences: {len(X_train)}, Val sequences: {len(X_val)}")
            return float('inf')

        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_class_train, dtype=torch.long),
            torch.tensor(y_reg_train, dtype=torch.float32).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_class_val, dtype=torch.long),
            torch.tensor(y_reg_val, dtype=torch.float32).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)
        logger.info(f"Trial {trial.number}: Train/Val DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    except Exception as e:
        logger.error(f"Trial {trial.number}: Error during data loading or splitting: {e}")
        return float('inf')


    # --- Mini-Training Loop for Optuna Trial ---
    EPOCHS_PER_TRIAL = 10 # Train for fewer epochs per trial to speed up tuning
    EARLY_STOPPING_PATIENCE_TRIAL = 3 # Patience within a trial

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(EPOCHS_PER_TRIAL):
        model.train()
        total_train_loss = 0
        total_train_class_loss = 0
        total_train_reg_loss = 0

        for batch_idx, (features, class_labels, reg_targets) in enumerate(train_loader):
            features, class_labels, reg_targets = features.to(device), class_labels.to(device), reg_targets.to(device)

            optimizer.zero_grad()
            class_logits, reg_output = model(features)

            class_loss = class_criterion(class_logits, class_labels)
            reg_loss = reg_criterion(reg_output, reg_targets)
            
            # Weighted combined loss
            loss = (classification_weight * class_loss) + (regression_weight * reg_loss)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
            optimizer.step()

            total_train_loss += loss.item()
            total_train_class_loss += class_loss.item()
            total_train_reg_loss += reg_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_class_loss = total_train_class_loss / len(train_loader)
        avg_train_reg_loss = total_train_reg_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0
        total_val_class_loss = 0
        total_val_reg_loss = 0
        all_val_preds_class = []
        all_val_true_class = []
        all_val_preds_reg = []
        all_val_true_reg = []

        with torch.no_grad():
            for features, class_labels, reg_targets in val_loader:
                features, class_labels, reg_targets = features.to(device), class_labels.to(device), reg_targets.to(device)

                class_logits, reg_output = model(features)

                val_class_loss = class_criterion(class_logits, class_labels)
                val_reg_loss = reg_criterion(reg_output, reg_targets)
                val_loss = (classification_weight * val_class_loss) + (regression_weight * val_reg_loss)

                total_val_loss += val_loss.item()
                total_val_class_loss += val_class_loss.item()
                total_val_reg_loss += val_reg_loss.item()

                _, predicted_classes = torch.max(class_logits, 1)
                all_val_preds_class.extend(predicted_classes.cpu().numpy())
                all_val_true_class.extend(class_labels.cpu().numpy())
                all_val_preds_reg.extend(reg_output.cpu().numpy())
                all_val_true_reg.extend(reg_targets.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_class_loss = total_val_class_loss / len(val_loader)
        avg_val_reg_loss = total_val_reg_loss / len(val_loader)
        
        # Calculate F1-score and RMSE for validation
        val_f1_weighted = f1_score(all_val_true_class, all_val_preds_class, average='weighted', zero_division=0)
        val_rmse = np.sqrt(mean_squared_error(all_val_true_reg, all_val_preds_reg))

        logger.info(f"Trial {trial.number}, Epoch {epoch+1}/{EPOCHS_PER_TRIAL}: "
                            f"Train Loss: {avg_train_loss:.4f} (Class: {avg_train_class_loss:.4f}, Reg: {avg_train_reg_loss:.4f}) | "
                            f"Val Loss: {avg_val_loss:.4f} (Class: {avg_val_class_loss:.4f}, Reg: {avg_val_reg_loss:.4f}) | "
                            f"Val F1: {val_f1_weighted:.4f}, Val RMSE: {val_rmse:.4f}")

        # Optuna pruning (early stopping for trials)
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            logger.info(f"Trial {trial.number} pruned at epoch {epoch+1}.")
            raise optuna.exceptions.TrialPruned()

        # Manual early stopping based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Optionally save the model for this trial if it's the best so far in this trial
            # torch.save(model.state_dict(), f"trial_{trial.number}_best_model.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE_TRIAL:
                logger.info(f"Trial {trial.number}: Early stopping triggered after {EARLY_STOPPING_PATIENCE_TRIAL} epochs without improvement.")
                break # Exit epoch loop for this trial

    logger.info(f"Trial {trial.number} finished. Best Val Loss: {best_val_loss:.4f}")
    return best_val_loss # Optuna minimizes this value

# --- Main execution of Optuna study ---
if __name__ == "__main__":
    logger.info("-" * 75)
    logger.info(" Aria-XaT Model Hyperparameter Tuning with Optuna")
    logger.info("-" * 75)

    # Optuna study creation
    # Direction: 'minimize' since we want to reduce the validation loss
    study_name = "aria_xat_hyperparam_tuning"
    storage_name = f"sqlite:///{study_name}.db" # Persist study results to a DB

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True # Continue from previous runs if DB exists
    )

    logger.info(f"Starting/Resuming Optuna study '{study_name}' with storage '{storage_name}'.")

    try:
        # Run the optimization for a specified number of trials
        # You can adjust n_trials based on your computational resources and desired exploration.
        # Start with a smaller number (e.g., 20-50) and then increase.
        study.optimize(objective, n_trials=50, show_progress_bar=True)
    except Exception as e:
        logger.error(f"Error during Optuna optimization: {e}")
        sys.exit(1)

    logger.info("\n--- Optuna Study Results ---")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Best trial:")
    trial = study.best_trial

    logger.info(f"   Value: {trial.value:.4f}")
    logger.info(f"   Params: ")
    for key, value in trial.params.items():
        logger.info(f"     {key}: {value}")

    # Optionally, save the best model found by Optuna in a dedicated file
    # This involves re-training the model with the best parameters on the full training data
    # (or loading the specific trial's saved model if you enabled that above).
    # For simplicity, we'll just print the best params. To save, you'd re-run the training process
    # using these best parameters and save the final model.

    logger.info("-" * 75)
    logger.info("Hyperparameter tuning complete.")
    logger.info("To use the best model, train your model from scratch using the best parameters found by Optuna.")
    logger.info("Or, if you enabled saving, load the model saved during the best trial.")