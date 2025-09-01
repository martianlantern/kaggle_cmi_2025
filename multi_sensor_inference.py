import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# -----------------------
# CONFIGURATION
# -----------------------
OUTPUT_DIR = "./multi_data_output"  # Where checkpoints and meta.json are saved
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# MODEL DEFINITION
# -----------------------
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, dropout: float = 0.2):
        super().__init__()
        def block(c_in, c_out, k=5, s=1, p=2):
            return nn.Sequential(
                nn.Conv1d(c_in, c_out, kernel_size=k, stride=s, padding=p, bias=False),
                nn.BatchNorm1d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2)
            )
        self.net = nn.Sequential(
            block(in_ch, 64),
            block(64, 128),
            block(128, 256, k=3, p=1),
            nn.Dropout(dropout),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.net(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x)

# -----------------------
# INFERENCE ENSEMBLE
# -----------------------
class InferenceEnsemble:
    def __init__(self, ckpt_paths, in_ch, n_classes, device=DEVICE, dropout=0.0):
        self.device = device
        self.models = []
        for p in ckpt_paths:
            m = CNN1D(in_ch, n_classes, dropout=dropout).to(device)
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

    def predict_label_for_sequence(self, x: np.ndarray) -> int:
        xb = torch.tensor(x[None, ...], dtype=torch.float32, device=self.device)
        probs = self.predict_proba_batch(xb)[0]
        return int(np.argmax(probs))

# -----------------------
# FEATURE PREPROCESSING
# -----------------------
def interpolate_fill_nan_safe(df, cols):
    df = df.copy()
    cols = [c for c in cols if c in df.columns]
    df[cols] = df[cols].interpolate(method="linear", limit_direction="both", axis=0).ffill().bfill()
    df[cols] = df[cols].fillna(0.0)
    return df

def resample_to_fixed_length(arr: np.ndarray, out_len: int) -> np.ndarray:
    T, C = arr.shape
    if T == out_len:
        return arr
    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, out_len)
    out = np.zeros((out_len, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = np.interp(x_new, x_old, arr[:, c])
    return out

def zscore_per_sequence(x: np.ndarray, eps=1e-6) -> np.ndarray:
    m = np.nanmean(x, axis=0)
    s = np.nanstd(x, axis=0)
    m = np.where(np.isnan(m), 0.0, m)
    s = np.where(np.isnan(s) | (s < eps), 1.0, s)
    return (x - m) / s

def prepare_sequence_features(seq_df, imu_cols, thm_cols=None, tof_sensors=5, tof_pixels=64, max_len=400, use_all_sensors=True):
    seq_df = interpolate_fill_nan_safe(seq_df, imu_cols)
    x_imu = seq_df[imu_cols].values.astype(np.float32)
    x_imu = resample_to_fixed_length(x_imu, max_len)
    x_imu = zscore_per_sequence(x_imu).T
    all_features = [x_imu]
    
    if use_all_sensors:
        # Thermopile
        if thm_cols:
            has_thm = any(c in seq_df.columns for c in thm_cols) and not seq_df[thm_cols].isna().all().all()
            if has_thm:
                seq_df = interpolate_fill_nan_safe(seq_df, thm_cols)
                x_thm = seq_df[thm_cols].values.astype(np.float32)
                x_thm = resample_to_fixed_length(x_thm, max_len)
                x_thm = zscore_per_sequence(x_thm).T
            else:
                x_thm = np.zeros((len(thm_cols), max_len), dtype=np.float32)
            all_features.append(x_thm)
        
        # ToF
        tof_features = []
        for sensor_id in range(1, tof_sensors + 1):
            tof_cols = [f"tof_{sensor_id}_v{i}" for i in range(tof_pixels)]
            existing_cols = [c for c in tof_cols if c in seq_df.columns]
            if existing_cols and not seq_df[existing_cols].isna().all().all():
                tof_data = seq_df[existing_cols].replace(-1, np.nan).mean(axis=1).fillna(0).values
                tof_data = tof_data.reshape(-1, 1)
                tof_data = resample_to_fixed_length(tof_data, max_len)
                tof_data = zscore_per_sequence(tof_data)
                tof_features.append(tof_data.T[0])
            else:
                tof_features.append(np.zeros(max_len, dtype=np.float32))
        if tof_features:
            all_features.append(np.stack(tof_features))
    
    x = np.concatenate(all_features, axis=0)
    return x

# -----------------------
# LOAD META AND ENSEMBLE
# -----------------------
meta = json.load(open(os.path.join(OUTPUT_DIR, "meta.json")))
ckpts = meta["fold_ckpts"]
imu_cols = meta["cfg"]["imu_cols"]
thm_cols = meta["cfg"]["thm_cols"]
tof_sensors = meta["cfg"]["tof_sensors"]
tof_pixels = meta["cfg"]["tof_pixels"]
max_len = meta["cfg"]["max_len"]
use_all_sensors = meta["cfg"]["use_all_sensors"]
idx_to_class = meta["classes"]

in_ch = len(imu_cols) + len(thm_cols) + tof_sensors if use_all_sensors else len(imu_cols)
ensemble = InferenceEnsemble(ckpts, in_ch, len(idx_to_class), device=DEVICE)

# -----------------------
# PREDICT ON TEST DATA
# -----------------------
test_data = pd.read_csv(os.path.join("./kaggle_cmi_2025/data", "test.csv"))
test_seq_ids = test_data["sequence_id"].unique()
preds = []

for seq_id in test_seq_ids:
    seq_df = test_data[test_data["sequence_id"] == seq_id].sort_values("sequence_counter")
    x = prepare_sequence_features(seq_df, imu_cols, thm_cols, tof_sensors, tof_pixels, max_len, use_all_sensors)
    pred_label = idx_to_class[ensemble.predict_label_for_sequence(x)]
    preds.append({"sequence_id": seq_id, "gesture": pred_label})

submission = pd.DataFrame(preds)
submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
print(f"Saved submission to: {os.path.join(OUTPUT_DIR, 'submission.csv')}")
