import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pandas as pd
import joblib
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, mean_squared_error
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Suppress warnings that might arise from pandas/numpy when converting to tensors
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor it is recommended to use sourceTensor.clone().detach()")

# --- Configuration for Training ---
config = {
    'processed_data_path': r'D:\aria\aria-xt-quant-pulse\aria\data\converted\processed_data_npy\All_Indices_XaT_enriched_ohlcv_indicators.joblib',
    'metadata_path': r'D:\aria\aria-xt-quant-pulse\aria\data\converted\models\chunk_metadata_xaat.pkl',
    'models_artifact_save_dir': r'D:\aria\aria-xt-quant-pulse\aria\data\converted\models',
    'best_model_filename': 'best_aria_xat_model.pth',
    'checkpoint_filename_prefix': 'aria_xat_checkpoint', # CHANGED: Removed redundant '_epoch'
    'train_ratio': 0.8,
    'val_ratio': 0.15,
    'batch_size': 256,
    'num_workers': 0, # Keep at 0 for Windows GPU for stability. Adjust only if data loading is bottleneck.
    'learning_rate': 0.001,
    'num_epochs': 50, # Increased epochs to allow for early stopping
    'classification_weight': 0.7, # Weight for classification loss in combined loss
    'regression_weight': 0.3, # Weight for regression loss in combined loss
    'early_stopping_patience': 10, # Number of epochs to wait for improvement
    'lr_scheduler_factor': 0.5, # Factor by which the learning rate will be reduced
    'lr_scheduler_patience': 5, # Number of epochs with no improvement after which learning rate will be reduced
    'SAVE_CHECKPOINT_EVERY_BATCHES': 2000, # Save checkpoint every X batches
    'num_checkpoints_to_keep': 5, # How many latest batch checkpoints to keep
    
    # --- Model Specific Parameters (NEWLY ADDED) ---
    'lstm_hidden_size': 128,
    'lstm_num_layers': 2,
    'num_classes': 3, # Assuming 3 classes for classification (e.g., up, down, flat)
    'dropout_rate': 0.2,
}

# Create model save directory if it doesn't exist
os.makedirs(config['models_artifact_save_dir'], exist_ok=True)

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Diagnostic for GPU ---
print(f"Attempting to use device: {device}")
if device.type == 'cuda':
    print(f"CUDA is available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("CUDA is NOT available despite being asked. Falling back to CPU. Check your PyTorch installation and CUDA drivers.")
else:
    print("CUDA is not available or not chosen. Using CPU.")


# --- Custom Dataset Class ---
class TimeSeriesDataset(Dataset):
    def __init__(self, processed_df, sequence_map, lookback_window, features, class_label_col, reg_label_col):
        self.processed_df = processed_df
        self.sequence_map = sequence_map
        self.lookback_window = lookback_window
        self.features = features
        self.class_label_col = class_label_col
        self.reg_label_col = reg_label_col

        # Pre-extract necessary columns into NumPy arrays for faster access in __getitem__
        # Ensure correct dtypes for PyTorch
        self.data_features = self.processed_df[self.features].values.astype(np.float32)
        self.class_labels = self.processed_df[self.class_label_col].values.astype(np.int64) # Long for CrossEntropyLoss
        self.reg_labels = self.processed_df[self.reg_label_col].values.astype(np.float32)

        # Build symbol_start_indices without using .get_loc(level='Symbol')
        # This relies on the DataFrame being sorted by Symbol first, which it is from preprocessing.
        self.symbol_start_indices = {}
        current_symbol = None
        # Iterate directly over the MultiIndex to find the start of each symbol's block
        for i, (symbol, _) in enumerate(self.processed_df.index): 
            if symbol != current_symbol:
                self.symbol_start_indices[symbol] = i
                current_symbol = symbol
        print(f"Built symbol_start_indices for {len(self.symbol_start_indices)} unique symbols.")

    def __len__(self):
        return len(self.sequence_map)

    def __getitem__(self, idx):
        # Each item in sequence_map is (symbol_name_str, relative_start_idx_in_symbol_df, relative_end_idx_in_symbol_df)
        symbol, relative_start_idx, relative_end_idx = self.sequence_map[idx]

        # Get the global starting index of this symbol's data block
        global_symbol_start_idx = self.symbol_start_indices[symbol]

        # Calculate the global start and end indices for the sequence
        global_sequence_start_idx = global_symbol_start_idx + relative_start_idx
        # global_sequence_end_idx is the index of the last point in the sequence, which is also its label's index
        global_sequence_end_idx = global_symbol_start_idx + relative_end_idx 

        # Extract data directly from the pre-extracted numpy arrays
        # Slice includes the end_idx, so +1 for Python slicing
        features_sequence = self.data_features[global_sequence_start_idx : global_sequence_end_idx + 1]
        
        # Labels are taken from the last point in the sequence
        class_label = self.class_labels[global_sequence_end_idx]
        reg_label = self.reg_labels[global_sequence_end_idx]

        # Convert to PyTorch tensors
        # features_sequence is already np.float32, so direct conversion is fine.
        return torch.from_numpy(features_sequence), \
               torch.tensor(class_label, dtype=torch.long), \
               torch.tensor(reg_label, dtype=torch.float32)

# --- Model Definition (CNN-LSTM-Attention) ---
class AriaXaTModel(nn.Module):
    def __init__(self, input_features, hidden_size=128, num_layers=2, output_classes=3, dropout_rate=0.2):
        super(AriaXaTModel, self).__init__()

        # CNN for feature extraction from each time step
        # Input: (batch_size, seq_len, input_features)
        # Permute to (batch_size, input_features, seq_len) for Conv1d
        self.conv1d = nn.Sequential(
            nn.Conv1d(input_features, 64, kernel_size=1), # kernel_size=1 processes each timestep independently
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, 32, kernel_size=1), # Reduce dimensions further
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # LSTM layer
        # Input to LSTM: (batch_size, seq_len, conv_output_features)
        self.lstm = nn.LSTM(input_size=32,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_rate if num_layers > 1 else 0)

        # Attention Mechanism
        self.attention_linear = nn.Linear(hidden_size, 1) # Maps LSTM output to a single score
        
        # Output layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, output_classes)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_features)
        
        # Apply Conv1d
        # Permute x to (batch_size, input_features, seq_len) for Conv1d
        x_conv = self.conv1d(x.permute(0, 2, 1))
        # Permute back to (batch_size, seq_len, conv_output_features) for LSTM
        x_conv = x_conv.permute(0, 2, 1) # x_conv shape: (batch_size, seq_len, 32)
        
        # Apply LSTM
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x_conv) 
        
        # Apply Attention
        # attention_scores shape: (batch_size, seq_len, 1)
        attention_scores = self.attention_linear(lstm_out)
        attention_weights = torch.softmax(attention_scores, dim=1) # Softmax over sequence length
        
        # Apply attention weights to LSTM output
        # context_vector shape: (batch_size, hidden_size)
        context_vector = torch.sum(lstm_out * attention_weights, dim=1)
        
        # Pass context vector to classifier and regressor
        classification_output = self.classifier(context_vector)
        regression_output = self.regressor(context_vector)
        
        return classification_output, regression_output.squeeze(1) # Squeeze for single regression value

# --- Checkpoint Management Function ---
# IMPROVED: Robustly parse epoch/batch from filename
def get_checkpoint_sort_key(filename):
    parts = filename.replace('.pth', '').split('_')
    epoch_val, batch_val = 0, 0
    for part in parts:
        if part.startswith('epoch'):
            try:
                epoch_val = int(part[len('epoch'):])
            except ValueError:
                pass # Malformed, will remain 0
        elif part.startswith('batch'):
            try:
                batch_val = int(part[len('batch'):])
            except ValueError:
                pass # Malformed, will remain 0
    
    # Warn only if it looks like a checkpoint but parsing failed to extract numbers
    if (epoch_val == 0 and batch_val == 0) and ('epoch' in filename or 'batch' in filename) and filename.startswith(config['checkpoint_filename_prefix']):
        warnings.warn(f"Malformed checkpoint filename encountered (parsing failed): {filename}. Defaulting to (0,0) for sorting.")
    return (epoch_val, batch_val)

def manage_checkpoints(save_dir, checkpoint_prefix, num_to_keep=5):
    all_checkpoints = [f for f in os.listdir(save_dir) if f.startswith(checkpoint_prefix) and 'epoch' in f and 'batch' in f]
    
    if len(all_checkpoints) <= num_to_keep:
        return

    checkpoint_info = []
    for cp_file in all_checkpoints:
        sort_key = get_checkpoint_sort_key(cp_file)
        # Only consider valid checkpoints for management
        if sort_key != (0, 0): # (0,0) usually means parsing failed or it's not a valid checkpoint filename
            checkpoint_info.append((sort_key, cp_file))

    # Sort oldest first (ascending order)
    checkpoint_info.sort(key=lambda x: x[0]) 

    # Keep the top 'num_to_keep' checkpoints (newest)
    # The [:-num_to_keep] slices out the oldest ones for deletion
    checkpoints_to_delete = [os.path.join(save_dir, info[1]) for info in checkpoint_info[:-num_to_keep]]

    # Delete the old checkpoints
    for cp_path in checkpoints_to_delete:
        try:
            os.remove(cp_path)
            # print(f"Deleted old checkpoint: {os.path.basename(cp_path)}") # Uncomment for verbose deletion
        except Exception as e:
            warnings.warn(f"Could not delete checkpoint {os.path.basename(cp_path)}: {e}")

# --- Training Function ---
def train_model():
    # Load preprocessed data and metadata
    print(f"\n--- Starting Model Training ---")
    print(f"Loading preprocessed data from: {config['processed_data_path']}")
    processed_df = joblib.load(config['processed_data_path'])
    print(f"Data loaded. Shape: {processed_df.shape}")

    print(f"Loading metadata from: {config['metadata_path']}")
    metadata = joblib.load(config['metadata_path'])
    print("Metadata loaded.")

    full_dataset = TimeSeriesDataset(
        processed_df=processed_df,
        sequence_map=metadata['sequence_map'],
        lookback_window=metadata['lookback_window'],
        features=metadata['features'],
        class_label_col=metadata['label_column'],
        reg_label_col=metadata['regression_label_column']
    )
    print(f"Dataset initialized with {len(full_dataset)} sequences.")

    # Split dataset into training, validation, and test sets
    total_sequences = len(full_dataset)
    train_size = int(config['train_ratio'] * total_sequences)
    val_size = int(config['val_ratio'] * total_sequences)
    test_size = total_sequences - train_size - val_size # The remainder goes to test

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    print(f"\nDataset split: Training sequences: {len(train_dataset)}, Validation sequences: {len(val_dataset)}, Test sequences: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        num_workers=config['num_workers'], 
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, # Do not shuffle validation data
        num_workers=config['num_workers'], 
        pin_memory=True if device.type == 'cuda' else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False, # Do not shuffle test data
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"\nDataLoader initialized with batch_size={config['batch_size']}, num_workers={config['num_workers']}.")
    print(f"Total training batches per epoch: {len(train_loader)}")
    print(f"Total validation batches per epoch: {len(val_loader)}")
    print(f"Total test batches: {len(test_loader)}")


    # Model, Loss, Optimizer
    input_features = metadata['num_features']
    model = AriaXaTModel(
        input_features=input_features,
        hidden_size=config['lstm_hidden_size'],
        num_layers=config['lstm_num_layers'],
        output_classes=config['num_classes'],
        dropout_rate=config['dropout_rate']
    ).to(device) # Move model to device

    print("\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {total_params}\n")

    criterion_class = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    # Scheduler monitors validation loss to reduce LR
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_scheduler_factor'], 
                                  patience=config['lr_scheduler_patience'])

    # Initialize GradScaler for Automatic Mixed Precision (AMP) if CUDA is available
    scaler_amp = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Load checkpoint if exists
    start_epoch = 1
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_epoch = 0
    latest_checkpoint_path = None

    # Find the latest checkpoint
    all_checkpoints = [f for f in os.listdir(config['models_artifact_save_dir']) if f.startswith(config['checkpoint_filename_prefix']) and 'epoch' in f and 'batch' in f]
    
    if all_checkpoints:
        # Filter out malformed filenames and get the latest valid one
        valid_checkpoints = []
        for cp_file in all_checkpoints:
            sort_key = get_checkpoint_sort_key(cp_file)
            if sort_key != (0, 0): # (0,0) indicates parsing failed, so skip
                valid_checkpoints.append((sort_key, cp_file))

        if valid_checkpoints:
            latest_checkpoint_filename = max(valid_checkpoints, key=lambda x: x[0])[1]
            latest_checkpoint_path = os.path.join(config['models_artifact_save_dir'], latest_checkpoint_filename)
            
            try:
                checkpoint = torch.load(latest_checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
                if 'scheduler_state_dict' in checkpoint and scheduler:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if scaler_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
                    scaler_amp.load_state_dict(checkpoint['scaler_state_dict'])

                print(f"Resuming training from checkpoint: {latest_checkpoint_filename} (Starting Epoch {start_epoch})")
            except Exception as e:
                print(f"Could not load checkpoint {latest_checkpoint_filename}: {e}. Starting training from scratch.")
        else:
            print("No valid checkpoints found after filtering, starting training from scratch.")
    else:
        print("No checkpoint found, starting training from scratch.")

    print(f"Starting training on {device} for {config['num_epochs']} epochs...")

    for epoch in range(start_epoch, config['num_epochs'] + 1):
        # --- Training Phase ---
        model.train()
        total_train_loss = 0.0
        total_train_class_loss = 0.0
        total_train_reg_loss = 0.0
        train_class_preds_list = []
        train_class_true_list = []
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{config['num_epochs']} (Train)")
        
        for batch_idx, (features, class_labels, reg_labels) in enumerate(train_loader_tqdm):
            features, class_labels, reg_labels = features.to(device), class_labels.to(device), reg_labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass: Use AMP for CUDA, regular forward for CPU
            if device.type == 'cuda':
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    class_output, reg_output = model(features)
                    loss_class = criterion_class(class_output, class_labels)
                    loss_reg = criterion_reg(reg_output, reg_labels)
                    loss = config['classification_weight'] * loss_class + config['regression_weight'] * loss_reg
            else: # CPU path
                class_output, reg_output = model(features)
                loss_class = criterion_class(class_output, class_labels)
                loss_reg = criterion_reg(reg_output, reg_labels)
                loss = config['classification_weight'] * loss_class + config['regression_weight'] * loss_reg

            # Backward pass and optimize with AMP if CUDA
            if device.type == 'cuda' and scaler_amp:
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer) # Unscale gradients before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else: # CPU path
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
                optimizer.step()

            total_train_loss += loss.item()
            total_train_class_loss += loss_class.item()
            total_train_reg_loss += loss_reg.item()
            
            train_class_preds_list.extend(class_output.argmax(dim=1).cpu().numpy())
            train_class_true_list.extend(class_labels.cpu().numpy())
            
            train_loader_tqdm.set_postfix(loss=f"{loss.item():.4f}", 
                                           class_loss=f"{loss_class.item():.4f}", 
                                           reg_loss=f"{loss_reg.item():.4f}",
                                           lr=f"{optimizer.param_groups[0]['lr']:.6f}")

            # Save checkpoint every X batches
            if (batch_idx + 1) % config['SAVE_CHECKPOINT_EVERY_BATCHES'] == 0:
                checkpoint_path = os.path.join(config['models_artifact_save_dir'], 
                                               f"{config['checkpoint_filename_prefix']}_epoch{epoch}_batch{batch_idx+1}.pth") # CHANGED: Filename format
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(), # Save last batch loss
                    'best_val_loss': best_val_loss,
                    'epochs_no_improve': epochs_no_improve,
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler_amp.state_dict() if scaler_amp else None,
                }, checkpoint_path)
                print(f"\n--- Checkpoint saved at batch {batch_idx+1} to {checkpoint_path} ---")
                manage_checkpoints(config['models_artifact_save_dir'], config['checkpoint_filename_prefix'], config['num_checkpoints_to_keep'])

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_class_loss = total_train_class_loss / len(train_loader)
        avg_train_reg_loss = total_train_reg_loss / len(train_loader)
        train_f1 = f1_score(train_class_true_list, train_class_preds_list, average='weighted', zero_division=0)

        print(f"\nEpoch {epoch} Training Summary:")
        print(f"   Avg Loss: {avg_train_loss:.4f}, Class Loss: {avg_train_class_loss:.4f}, Reg Loss: {avg_train_reg_loss:.4f}")
        print(f"   Train F1-Score (Weighted): {train_f1:.4f}")

        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        total_val_class_loss = 0.0
        total_val_reg_loss = 0.0
        val_class_preds_list = []
        val_class_true_list = []
        val_reg_preds_list = []
        val_reg_true_list = []

        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch}/{config['num_epochs']} (Validation)")
            for features, class_labels, reg_labels in val_loader_tqdm:
                features, class_labels, reg_labels = features.to(device), class_labels.to(device), reg_labels.to(device)

                # Forward pass: Use AMP for CUDA, regular forward for CPU
                if device.type == 'cuda':
                    with torch.autocast(device_type=device.type, dtype=torch.float16):
                        class_output, reg_output = model(features)
                        loss_class = criterion_class(class_output, class_labels)
                        loss_reg = criterion_reg(reg_output, reg_labels)
                        val_loss = config['classification_weight'] * loss_class + config['regression_weight'] * loss_reg
                else: # CPU path
                    class_output, reg_output = model(features)
                    loss_class = criterion_class(class_output, class_labels)
                    loss_reg = criterion_reg(reg_output, reg_labels)
                    val_loss = config['classification_weight'] * loss_class + config['regression_weight'] * loss_reg
                
                total_val_loss += val_loss.item()
                total_val_class_loss += loss_class.item()
                total_val_reg_loss += loss_reg.item()

                val_class_preds_list.extend(class_output.argmax(dim=1).cpu().numpy())
                val_class_true_list.extend(class_labels.cpu().numpy())
                val_reg_preds_list.extend(reg_output.cpu().numpy())
                val_reg_true_list.extend(reg_labels.cpu().numpy())

                val_loader_tqdm.set_postfix(val_loss=f"{val_loss.item():.4f}", 
                                             val_class_loss=f"{loss_class.item():.4f}", 
                                             val_reg_loss=f"{loss_reg.item():.4f}")

        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_class_loss = total_val_class_loss / len(val_loader)
        avg_val_reg_loss = total_val_reg_loss / len(val_loader)
        val_f1 = f1_score(val_class_true_list, val_class_preds_list, average='weighted', zero_division=0)
        val_rmse = np.sqrt(mean_squared_error(val_reg_true_list, val_reg_preds_list))

        print(f"Epoch {epoch} Validation Summary:")
        print(f"   Avg Val Loss: {avg_val_loss:.4f}, Class Loss: {avg_val_class_loss:.4f}, Reg Loss: {avg_val_reg_loss:.4f}")
        print(f"   Val F1-Score (Weighted): {val_f1:.4f}, Val RMSE: {val_rmse:.4f}")

        # Learning Rate Scheduler step
        scheduler.step(avg_val_loss)

        # Early Stopping and Best Model Saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_epoch = epoch
            model_save_path = os.path.join(config['models_artifact_save_dir'], config['best_model_filename'])
            torch.save(model.state_dict(), model_save_path)
            print(f"--- New best model saved at Epoch {best_epoch} with Validation Loss: {best_val_loss:.4f} ---")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs. Best Val Loss: {best_val_loss:.4f}")
            if epochs_no_improve >= config['early_stopping_patience']:
                print(f"Early stopping triggered after {epochs_no_improve} epochs without improvement.")
                print(f"Best model was saved at Epoch {best_epoch} with Validation Loss: {best_val_loss:.4f}")
                break # Exit training loop

    print("\n--- Training Complete ---")
    
    # --- Final Test Evaluation ---
    print("\nEvaluating on Test Set...")
    # It's good practice to load the best model for final evaluation
    best_model_path = os.path.join(config['models_artifact_save_dir'], config['best_model_filename'])
    if os.path.exists(best_model_path):
        print(f"Loading best model for final test evaluation: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Best model not found for test evaluation. Using the last trained model state.")

    model.eval()
    test_total_loss = 0.0
    test_total_class_loss = 0.0
    test_total_reg_loss = 0.0
    test_class_preds_list = []
    test_class_true_list = []
    test_reg_preds_list = []
    test_reg_true_list = []

    with torch.no_grad():
        for features, class_labels, reg_labels in tqdm(test_loader, desc="Test Evaluation"):
            features, class_labels, reg_labels = features.to(device), class_labels.to(device), reg_labels.to(device)

            # Forward pass: Use AMP for CUDA, regular forward for CPU
            if device.type == 'cuda':
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    class_output, reg_output = model(features)
                    loss_class = criterion_class(class_output, class_labels)
                    loss_reg = criterion_reg(reg_output, reg_labels)
                    test_loss = config['classification_weight'] * loss_class + config['regression_weight'] * loss_reg
            else: # CPU path
                class_output, reg_output = model(features)
                loss_class = criterion_class(class_output, class_labels)
                loss_reg = criterion_reg(reg_output, reg_labels)
                test_loss = config['classification_weight'] * loss_class + config['regression_weight'] * loss_reg
            
            test_total_loss += test_loss.item()
            test_total_class_loss += loss_class.item()
            test_total_reg_loss += loss_reg.item()

            test_class_preds_list.extend(class_output.argmax(dim=1).cpu().numpy())
            test_class_true_list.extend(class_labels.cpu().numpy())
            test_reg_preds_list.extend(reg_output.cpu().numpy())
            test_reg_true_list.extend(reg_labels.cpu().numpy())

    avg_test_loss = test_total_loss / len(test_loader)
    avg_test_class_loss = test_total_class_loss / len(test_loader)
    avg_test_reg_loss = test_total_reg_loss / len(test_loader)
    test_f1 = f1_score(test_class_true_list, test_class_preds_list, average='weighted', zero_division=0)
    test_rmse = np.sqrt(mean_squared_error(test_reg_true_list, test_reg_preds_list))

    print(f"\n--- Test Set Evaluation Results ---")
    print(f"   Avg Test Loss: {avg_test_loss:.4f} (Class: {avg_test_class_loss:.4f}, Reg: {avg_test_reg_loss:.4f})")
    print(f"   Test F1-Score (Weighted): {test_f1:.4f}, Test RMSE: {test_rmse:.4f}")
    print(f"--- All processes completed for Aria-XaT Model Training ---")


if __name__ == '__main__':
    train_model()