import joblib
import os

# Path to the metadata file (update if your path is different)
metadata_path = r'D:\aria\aria-xt-quant-pulse\aria\data\converted\models\chunk_metadata_xaat.pkl'

if not os.path.exists(metadata_path):
    print(f"Metadata file not found: {metadata_path}")
    exit(1)

metadata = joblib.load(metadata_path)

features = metadata.get('features')
num_features = metadata.get('num_features')
lookback_window = metadata.get('lookback_window')

print("\n--- Aria-XaT Model Input Features ---")
if features:
    for i, feat in enumerate(features):
        print(f"{i+1:2d}. {feat}")
else:
    print("No features found in metadata.")

print(f"\nTotal features: {num_features}")
print(f"Lookback window: {lookback_window}")
