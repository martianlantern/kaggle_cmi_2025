import os
import json
import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from tqdm.auto import tqdm

OUTPUT_DIR = "./outputs"
TEST_DATA_PATH = "./data/test.csv"
TEST_DEMOGRAPHICS_PATH = "./data/test_demographics.csv"
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# LSTM Model Definition (same as in 4_LSTM_THM.py)
class LSTM(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=in_ch,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=False,
            bidirectional=True
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        x = x.transpose(0, 1)
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[-1]
        out = self.dropout(last_output)
        return self.head(out)

# Data preprocessing functions (same as in 4_LSTM_THM.py)
def interpolate_fill_nan_safe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
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

def resample_to_fixed_length(arr: np.ndarray, out_len: int) -> np.ndarray:
    """Resample T×C to out_len×C via per-channel linear interpolation in [0,1]."""
    T, C = arr.shape
    if T == out_len:
        return arr
    if T <= 1:
        # If degenerate length, just repeat or pad zeros
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
    Channel-wise z-score per sequence, robust when an entire channel is constant or NaN.
    x: T×C
    """
    m = np.nanmean(x, axis=0)
    s = np.nanstd(x, axis=0)
    # Replace NaN means with 0; replace tiny/NaN std with 1 (so output becomes ~0)
    m = np.where(np.isnan(m), 0.0, m)
    s = np.where(np.isnan(s) | (s < eps), 1.0, s)
    return (x - m) / s

# Inference Ensemble
class InferenceEnsemble:
    def __init__(self, ckpt_paths: List[str], in_ch: int, n_classes: int, device=DEVICE, dropout=0.0):
        self.device = device
        self.models = []
        for p in ckpt_paths:
            m = LSTM(in_ch, n_classes, hidden_size=128, num_layers=2, dropout=dropout).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            m.eval()
            self.models.append(m)

    @torch.no_grad()
    def predict_proba_batch(self, xb: torch.Tensor) -> np.ndarray:
        probs = []
        for m in self.models:
            logits = m(xb)
            probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.mean(probs, axis=0)

    def predict_label_for_sequence_df(self, seq_df: pd.DataFrame) -> str:
        """
        Predict gesture label for a single sequence DataFrame.
        """
        # Get the metadata to access configuration
        ensemble, meta = _get_ensemble()
        cfg = meta['cfg']
        imu_cols = cfg['imu_cols']
        thermopile_cols = cfg['thermopile_cols']
        max_len = cfg['max_len']
        
        # Combine all sensor columns
        all_cols = list(imu_cols) + list(thermopile_cols)
        
        # Ensure missing columns are present for pipeline consistency
        for col in all_cols:
            if col not in seq_df.columns:
                seq_df[col] = 0.0
        
        # Check if it's already a pandas DataFrame, if not convert it
        if hasattr(seq_df, 'to_pandas'):
            # It's a polars DataFrame, convert to pandas
            seq_df = seq_df.to_pandas()
        # If it's already pandas, use it as is
        
        # Apply the EXACT same preprocessing pipeline as training
        seq_df = interpolate_fill_nan_safe(seq_df, all_cols)
        x = seq_df[all_cols].astype(np.float32).values  # (T, C)
        x = resample_to_fixed_length(x, max_len)  # (max_len, C)
        x = zscore_per_sequence(x)  # (max_len, C) - keep as (L, C) to match training
        
        # Convert to tensor with shape (1, L, C) to match training batch format
        xb = torch.tensor(x[None, ...], dtype=torch.float32, device=self.device)
        
        # Make prediction using the ensemble
        probs = self.predict_proba_batch(xb)[0]
        pred_idx = int(np.argmax(probs))
        return meta['classes'][pred_idx]

# Global variables for model loading
_ENSEMBLE = None
_meta = None

def _get_ensemble():
    """Get or create the ensemble model (lazy loading)"""
    global _ENSEMBLE, _meta
    if _ENSEMBLE is None:
        # Load metadata
        meta_path = os.path.join(OUTPUT_DIR, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found at {meta_path}")
        
        with open(meta_path, 'r') as f:
            _meta = json.load(f)
        
        # Load ensemble
        ckpt_paths = _meta['fold_ckpts']
        cfg = _meta['cfg']
        in_ch = len(cfg['imu_cols']) + len(cfg['thermopile_cols'])
        n_classes = len(_meta['classes'])
        
        _ENSEMBLE = InferenceEnsemble(ckpt_paths, in_ch, n_classes, device=DEVICE, dropout=0.0)
        print(f"Loaded {len(ckpt_paths)} models for inference")
        print(f"Device: {DEVICE}")
        print(f"Input channels: {in_ch} (IMU: {len(cfg['imu_cols'])}, Thermopile: {len(cfg['thermopile_cols'])})")
    
    return _ENSEMBLE, _meta

def predict(sequence_df: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Kaggle will call this repeatedly, one sequence at a time.
    Input: DataFrame with IMU and thermopile columns.
    Returns: gesture label as string.
    """
    try:
        # Get the loaded ensemble and metadata
        ensemble, meta = _get_ensemble()
        
        # Extract configuration
        cfg = meta['cfg']
        imu_cols = cfg['imu_cols']
        thermopile_cols = cfg['thermopile_cols']
        max_len = cfg['max_len']
        idx_to_class = meta['classes']
        
        # Combine all sensor columns
        all_cols = list(imu_cols) + list(thermopile_cols)
        
        # Ensure missing columns are present for pipeline consistency
        for col in all_cols:
            if col not in sequence_df.columns:
                sequence_df = sequence_df.with_columns(pl.lit(0.0).alias(col))
        
        # Convert polars DataFrame to pandas for compatibility with existing pipeline
        seq_df = sequence_df.to_pandas()
        
        # Make prediction using the ensemble
        return ensemble.predict_label_for_sequence_df(seq_df)
    
    except Exception as e:
        # Default prediction if anything fails
        print(f"Prediction failed with error: {e}. Returning default prediction: 'text on phone'")
        return "Text on phone"

def test_inference():
    """Test the inference pipeline with actual test data"""
    print("Testing inference pipeline...")
    
    # Load test data
    try:
        test_data = pd.read_csv(TEST_DATA_PATH)
        test_demographics = pd.read_csv(TEST_DEMOGRAPHICS_PATH)
        print(f"Loaded test data: {len(test_data)} rows")
        print(f"Loaded demographics: {len(test_demographics)} rows")
    except FileNotFoundError as e:
        print(f"Error loading test data: {e}")
        print("Make sure the test data files exist at the specified paths")
        return
    
    # Get unique sequence IDs
    sequence_ids = test_data['sequence_id'].unique()
    print(f"Found {len(sequence_ids)} unique sequences")
    
    # Test inference on first few sequences
    test_sequences = sequence_ids[:5]  # Test first 5 sequences
    predictions = []
    
    for seq_id in tqdm(test_sequences, desc="Testing inference"):
        # Get sequence data
        seq_data = test_data[test_data['sequence_id'] == seq_id].sort_values('sequence_counter')
        
        # Convert to polars
        seq_df = pl.DataFrame(seq_data)
        demo_df = pl.DataFrame(test_demographics[test_demographics['subject'] == seq_data['subject'].iloc[0]])
        
        try:
            # Make prediction
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
    
    # Summary
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

# Import and setup inference server
import kaggle_evaluation.cmi_inference_server

if __name__ == "__main__":
    # Test inference first
    test_results = test_inference()
    
    # Then setup inference server
    print("\n" + "="*50)
    print("SETTING UP INFERENCE SERVER")
    print("="*50)
    
    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

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
