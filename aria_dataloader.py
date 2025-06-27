import os
import pandas as pd
import numpy as np
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
import warnings

# Suppress warnings that might arise from pandas/numpy when converting to tensors
warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor it is recommended to use sourceTensor.clone().detach()")

# --- Configuration (matching your preprocessing script's output paths) ---
# IMPORTANT: Adjust these paths if your project structure changes
config = {
    'processed_data_dir': r'D:\aria\aria-xt-quant-pulse\aria\data\converted\processed_data_npy',
    'models_artifact_dir': r'D:\aria\aria-xt-quant-pulse\aria\data\converted\models',
    'processed_filename': 'All_Indices_XaT_enriched_ohlcv_indicators.joblib',
    'metadata_filename': 'chunk_metadata_xaat.pkl',
    'lookback_window': 60, # Must match the lookback_window used during preprocessing
}

class AriaXaTDataset(Dataset):
    """
    Custom PyTorch Dataset for Aria-XaT quant-pulse model.
    Loads preprocessed multi-indexed data and provides sequences
    for CNN-LSTM-Attention model training with dual outputs.
    """
    def __init__(self, data_dir, processed_filename, metadata_filename):
        """
        Initializes the dataset by loading the preprocessed data and metadata.

        Args:
            data_dir (str): Directory where the processed data (.joblib) is saved.
            processed_filename (str): Name of the processed data file.
            metadata_filename (str): Name of the metadata file.
        """
        self.data_filepath = os.path.join(data_dir, processed_filename)
        self.metadata_filepath = os.path.join(data_dir.replace('processed_data_npy', 'models'), metadata_filename) # Correct path to models dir

        if not os.path.exists(self.data_filepath):
            raise FileNotFoundError(f"Processed data file not found: {self.data_filepath}")
        if not os.path.exists(self.metadata_filepath):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_filepath}")

        print(f"Loading preprocessed data from: {self.data_filepath}")
        self.data_df = joblib.load(self.data_filepath)
        print(f"Data loaded. Shape: {self.data_df.shape}")

        print(f"Loading metadata from: {self.metadata_filepath}")
        self.metadata = joblib.load(self.metadata_filepath)
        print("Metadata loaded.")

        self.sequence_map = self.metadata['sequence_map']
        self.lookback_window = self.metadata['lookback_window']
        self.features = self.metadata['features']
        self.label_column = self.metadata['label_column']
        self.regression_label_column = self.metadata['regression_label_column']

        # Ensure all required columns are in the DataFrame
        missing_features = [f for f in self.features if f not in self.data_df.columns]
        if missing_features:
            raise ValueError(f"Missing features in loaded DataFrame: {missing_features}")
        if self.label_column not in self.data_df.columns:
            raise ValueError(f"Missing classification label column: {self.label_column}")
        if self.regression_label_column not in self.data_df.columns:
            raise ValueError(f"Missing regression label column: {self.regression_label_column}")

        print(f"Dataset initialized with {len(self.sequence_map)} sequences.")

    def __len__(self):
        """
        Returns the total number of sequences in the dataset.
        """
        return len(self.sequence_map)

    def __getitem__(self, idx):
        """
        Retrieves a single sequence (X, y_class, y_reg) by index.

        Args:
            idx (int): The index of the sequence to retrieve from the sequence_map.

        Returns:
            tuple: (features_tensor, class_label_tensor, regression_label_tensor)
        """
        if not (0 <= idx < len(self.sequence_map)):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self.sequence_map)}")

        symbol, start_idx_in_symbol_df, end_idx_in_symbol_df = self.sequence_map[idx]

        # Efficiently slice the sub-DataFrame for the specific symbol
        # .loc[symbol] gives a DataFrame for that symbol, then .iloc[start:end+1]
        # to get the exact lookback window.
        symbol_df_slice = self.data_df.loc[symbol].iloc[start_idx_in_symbol_df : end_idx_in_symbol_df + 1]

        # Extract features (X)
        # Ensure correct order of features as defined in metadata
        features_np = symbol_df_slice[self.features].values

        # Extract labels (y_class and y_reg)
        # The label for a sequence is the label at the *end* of the sequence window (end_idx_in_symbol_df).
        # This aligns with predicting the next future target based on the information up to end_idx.
        class_label_np = symbol_df_slice[self.label_column].iloc[-1]
        regression_label_np = symbol_df_slice[self.regression_label_column].iloc[-1]

        # Convert to PyTorch tensors
        # Features should be float32, labels can be long (for classification) or float32 (for regression)
        features_tensor = torch.from_numpy(features_np).float()
        class_label_tensor = torch.tensor(class_label_np).long() # Long for classification targets
        regression_label_tensor = torch.tensor(regression_label_np).float() # Float for regression targets

        return features_tensor, class_label_tensor, regression_label_tensor

# --- Example Usage (main block) ---
if __name__ == '__main__':
    print("--- Starting Dataset and DataLoader Demonstration ---")

    # 1. Instantiate the Dataset
    try:
        aria_dataset = AriaXaTDataset(
            data_dir=config['processed_data_dir'],
            processed_filename=config['processed_filename'],
            metadata_filename=config['metadata_filename']
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure your data preprocessing script has run successfully and paths are correct.")
        exit()
    except ValueError as e:
        print(f"Data loading error: {e}")
        exit()

    # 2. Instantiate the DataLoader
    # Adjust batch_size and num_workers based on your system's capabilities
    # For initial testing, small batch_size and num_workers=0 (main process) is good.
    # Increase num_workers for faster loading during actual training if data loading is a bottleneck.
    # Note: num_workers > 0 might not be efficient if your data is on a slow HDD or if joblib.load is bottlenecking.
    # For a single large joblib file, num_workers=0 might sometimes be faster than multiple workers competing for disk I/O.
    batch_size = 32
    num_workers = 0 # Start with 0 (main process) for debugging. Set to os.cpu_count() or fewer for speed.

    aria_dataloader = DataLoader(
        aria_dataset,
        batch_size=batch_size,
        shuffle=True, # Shuffle data for training
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False # Speeds up data transfer to GPU
    )

    print(f"\nDataLoader initialized with batch_size={batch_size}, num_workers={num_workers}.")
    print(f"Total batches per epoch: {len(aria_dataloader)}")

    # 3. Iterate through a few batches to verify
    print("\nVerifying data batches...")
    for i, (features, class_labels, regression_labels) in enumerate(aria_dataloader):
        print(f"\nBatch {i+1}:")
        print(f"  Features shape: {features.shape} (Batch Size, Lookback Window, Num Features)")
        print(f"  Class Labels shape: {class_labels.shape} (Batch Size)")
        print(f"  Regression Labels shape: {regression_labels.shape} (Batch Size)")

        # Optional: Print some values from the first item in the batch
        print(f"  Example Feature sequence (first item in batch, first 5 features):")
        print(features[0, :, :5])
        print(f"  Example Class Label (first item in batch): {class_labels[0].item()}")
        print(f"  Example Regression Label (first item in batch): {regression_labels[0].item():.4f}")

        if i >= 2: # Stop after 3 batches for demonstration
            break
    
    print("\n--- Dataset and DataLoader Demonstration Complete ---")
    print("\nYour data is now ready to be fed into your PyTorch model for training!")