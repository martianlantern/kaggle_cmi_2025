import os
import json
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from tqdm.auto import tqdm

# Paths to test data and demographics
TEST_DATA_PATH = "./data/test.csv"
TEST_DEMOGRAPHICS_PATH = "./data/test_demographics.csv"

# Device selection for PyTorch (MPS, CUDA, or CPU)
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# Define an LSTM model class for sequence prediction
class LSTM(nn.Module):
    """
    LSTM-based neural network for sequence classification.
    """
    def __init__(self, in_ch: int, n_classes: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer with bidirectional support
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False,  # Input shape: (L, B, C)
            bidirectional=True
        )
        
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)
        # Linear layer to map LSTM outputs to class predictions
        self.head = nn.Linear(hidden_size * 2, n_classes)  # *2 for bidirectional

    def forward(self, x):
        # Transpose input to match LSTM expectations: (B, L, C) -> (L, B, C)
        x = x.transpose(0, 1)
        # Forward pass through LSTM
        lstm_out, _ = self.lstm(x)
        # Use the last output of LSTM (last time step)
        last_output = lstm_out[-1]
        # Apply dropout and pass through the linear layer
        out = self.dropout(last_output)
        out = self.head(out)
        return out

# Function to interpolate and fill NaN values in a DataFrame
def interpolate_fill_nan_safe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Interpolates and fills NaN values in the specified columns of a DataFrame.
    Ensures no column remains all-NaN; fills with 0 if so.
    """
    out = df.copy()
    cols = [c for c in cols if c in out.columns]
    out[cols] = out[cols].astype(np.float32)
    out[cols] = out[cols].interpolate(method="linear", limit_direction="both", axis=0)
    out[cols] = out[cols].ffill().bfill()
    still_all_nan = out[cols].isna().all(axis=0)
    if still_all_nan.any():
        for c in out[cols].columns[still_all_nan]:
            out[c] = 0.0
    med = out[cols].median(axis=0)
    out[cols] = out[cols].fillna(med).fillna(0.0)
    return out

# Function to resample an array to a fixed length
def resample_to_fixed_length(arr: np.ndarray, out_len: int) -> np.ndarray:
    """
    Resamples a 2D array (T, C) to a fixed number of time steps (out_len).
    Uses linear interpolation for each channel.
    """
    T, C = arr.shape
    if T == out_len:
        return arr
    if T <= 1:
        base = np.zeros((out_len, C), dtype=np.float32)
        if T == 1:
            base[:] = arr[0]
        return base
    x_old = np.linspace(0.0, 1.0, num=T, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, num=out_len, dtype=np.float32)
    out = np.empty((out_len, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = np.interp(x_new, x_old, arr[:, c])
    return out

def zscore_per_sequence(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Standardizes each channel of the sequence to zero mean and unit variance.
    Handles NaNs robustly.
    """
    m = np.nanmean(x, axis=0)
    s = np.nanstd(x, axis=0)
    m = np.where(np.isnan(m), 0.0, m)
    s = np.where(np.isnan(s) | (s < eps), 1.0, s)
    return (x - m) / s

class InferenceEnsemble:
    """
    Ensemble of LSTM models for robust sequence classification.
    """
    def __init__(self, ckpt_paths: List[str], in_ch: int, n_classes: int, device=DEVICE, dropout=0.0):
        self.device = device
        self.models = []
        # Load each model checkpoint and set to eval mode
        for p in ckpt_paths:
            m = LSTM(in_ch, n_classes, hidden_size=128, num_layers=2, dropout=dropout).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            m.eval()
            self.models.append(m)

    @torch.no_grad()
    def predict_proba_batch(self, xb: torch.Tensor) -> np.ndarray:
        """
        Predicts class probabilities for a batch using the ensemble.
        Returns the mean probability across all models.
        """
        probs = []
        for m in self.models:
            logits = m(xb)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.mean(probs, axis=0)

    def predict_label_for_sequence_df(self, seq_df: pd.DataFrame) -> str:
        """
        Given a sequence DataFrame, preprocesses and predicts the gesture label.
        """
        ensemble, meta = _get_ensemble()
        cfg = meta['cfg']
        imu_cols = cfg['imu_cols']
        max_len = cfg['max_len']
        
        # Ensure all required IMU columns are present
        for col in imu_cols:
            if col not in seq_df.columns:
                seq_df[col] = 0.0
        
        # Convert to pandas DataFrame if needed
        if hasattr(seq_df, 'to_pandas'):
            seq_df = seq_df.to_pandas()
        
        # Preprocess: interpolate/fill NaNs, resample, z-score
        seq_df = interpolate_fill_nan_safe(seq_df, imu_cols)
        x = seq_df[imu_cols].astype(np.float32).values
        x = resample_to_fixed_length(x, max_len)
        x = zscore_per_sequence(x)
        
        # Prepare tensor for model input
        xb = torch.tensor(x[None, ...], dtype=torch.float32, device=self.device)
        
        # Get ensemble probabilities and predicted class
        probs = self.predict_proba_batch(xb)[0]
        pred_idx = int(np.argmax(probs))
        return meta['classes'][pred_idx]

# Global variables for lazy loading of ensemble and metadata
_ENSEMBLE = None
_meta = None

def _get_ensemble():
    """
    Loads the ensemble and metadata if not already loaded.
    Returns the ensemble and metadata.
    """
    global _ENSEMBLE, _meta
    if _ENSEMBLE is None:
        meta_path = os.path.join(OUTPUT_DIR, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        
        with open(meta_path, 'r') as f:
            _meta = json.load(f)
        
        ckpt_paths = _meta['fold_ckpts']
        cfg = _meta['cfg']
        in_ch = len(cfg['imu_cols'])
        n_classes = len(_meta['classes'])
        
        _ENSEMBLE = InferenceEnsemble(ckpt_paths, in_ch, n_classes, device=DEVICE, dropout=0.0)
        print(f"Loaded {len(ckpt_paths)} models for inference")
        print(f"Device: {DEVICE}")
    
    return _ENSEMBLE, _meta

def predict(sequence_df: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Main prediction function for the inference server.
    Accepts a Polars DataFrame for the sequence and demographics.
    Returns the predicted gesture label as a string.
    """
    try:
        ensemble, meta = _get_ensemble()
        cfg = meta['cfg']
        imu_cols = cfg['imu_cols']
        max_len = cfg['max_len']
        idx_to_class = meta['classes']
        
        # Ensure all IMU columns are present in the input
        for col in imu_cols:
            if col not in sequence_df.columns:
                sequence_df[col] = 0.0
        
        # Convert to pandas DataFrame for processing
        seq_df = sequence_df.to_pandas()
        
        return ensemble.predict_label_for_sequence_df(seq_df)
    
    except Exception as e:
        print(f"Prediction failed with error: {e}. Returning default prediction: 'text on phone'")
        return "Text on phone"

def test_inference():
    """
    Test the inference pipeline on a few sequences from the test set.
    Prints summary statistics and sample predictions.
    """
    print("Testing inference pipeline...")
    
    try:
        test_data = pd.read_csv(TEST_DATA_PATH)
        test_demographics = pd.read_csv(TEST_DEMOGRAPHICS_PATH)
        print(f"Loaded test data: {len(test_data)} rows")
        print(f"Loaded demographics: {len(test_demographics)} rows")
    except FileNotFoundError as e:
        print(f"Error loading test data: {e}")
        print("Make sure the test data files exist at the specified paths")
        return
    
    sequence_ids = test_data['sequence_id'].unique()
    print(f"Found {len(sequence_ids)} unique sequences")
    
    # Only test on a small subset for speed
    test_sequences = sequence_ids[:5]
    predictions = []
    
    for seq_id in tqdm(test_sequences, desc="Testing inference"):
        # Extract sequence data and sort by time
        seq_data = test_data[test_data['sequence_id'] == seq_id].sort_values('sequence_counter')
        
        seq_df = pl.DataFrame(seq_data)
        demo_df = pl.DataFrame(test_demographics[test_demographics['subject'] == seq_data['subject'].iloc[0]])
        
        try:
            pred = predict(seq_df, demo_df)
            predictions.append({
                'sequence_id': seq_id,
                'predicted_gesture': pred,
                'sequence_length': len(seq_data)
            })
            print(f"Sequence {seq_id}: Predicted '{pred}' (length: {len(seq_data)})")
        except Exception as e:
            print(f"Error predicting sequence {seq_id}: {e}")
            predictions.append({
                'sequence_id': seq_id,
                'predicted_gesture': 'ERROR',
                'sequence_length': len(seq_data)
            })
    
    # Print summary of test results
    print("\n" + "="*50)
    print("INFERENCE TEST SUMMARY")
    print("="*50)
    print(f"Tested {len(predictions)} sequences")
    successful_predictions = [p for p in predictions if p['predicted_gesture'] != 'ERROR']
    print(f"Successful predictions: {len(successful_predictions)}")
    print(f"Failed predictions: {len(predictions) - len(successful_predictions)}")
    
    if successful_predictions:
        print("\nSample predictions:")
        for pred in successful_predictions[:20]:
            print(f"  {pred['sequence_id']}: {pred['predicted_gesture']}")
    
    return str(predictions)

# Import the inference server module for Kaggle evaluation
import kaggle_evaluation.cmi_inference_server

if __name__ == "__main__":
    # Run a quick test of the inference pipeline
    test_results = test_inference()
    
    print("\n" + "="*50)
    print("SETTING UP INFERENCE SERVER")
    print("="*50)
    
    # Initialize the inference server with the predict function
    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

    # Serve the inference server depending on the environment
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        print("Running in competition mode - serving inference server...")
        inference_server.serve()
    else:
        print("Running in local mode - starting local gateway...")
        inference_server.run_local_gateway(
            data_paths=(
                TEST_DATA_PATH,
                TEST_DEMOGRAPHICS_PATH,
            )
        )