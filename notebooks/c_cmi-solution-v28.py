# TestConfig
# CV macroF1=0.2003 | binF1=0.8451 | meanF1=0.5227

# %%
import os
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import polars as pl

from tqdm.auto import tqdm

from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import StandardScaler

# %%
INPUT_DIR = "../data"
OUTPUT_DIR = "../outputs/cmi-solution-v28"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
def set_seed(seed=RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed()

DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
# %%
import time
start_time = time.time()
print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")


# %%
@dataclass
class Config:
    max_len: int = 400
    batch_size: int = 64
    num_workers: int = 8
    n_folds: int = 5
    n_epochs: int = 12
    lr: float = 2e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    imu_cols: Tuple[str, ...] = ("acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z")
    thm_cols: Tuple[str, ...] = ("thm_1", "thm_2", "thm_3", "thm_4", "thm_5")
    tof_sensors: int = 5  # 5 ToF sensors with 64 pixels each
    tof_pixels: int = 64  # 8x8 grid per sensor
    early_stopping_patience: int = 4
    grad_clip_norm: float = 1.0
    verbose: int = 1
    kaggle: bool = False
    use_all_sensors: bool = True  # Flag to enable multi-sensor fusion
    quick_test = False

@dataclass
class TestConfig:
    max_len: int = 100          # Reduce from 400 (faster data processing)
    batch_size: int = 16        # Reduce from 64 (smaller memory footprint)
    num_workers: int = 1        # Reduce from 8 (fewer multiprocessing issues)
    n_folds: int = 2            # Reduce from 5 (faster cross-validation)
    n_epochs: int = 3           # Reduce from 12 (much faster training)
    lr: float = 2e-3           # Keep same
    weight_decay: float = 1e-4  # Keep same
    dropout: float = 0.2        # Keep same
    imu_cols: Tuple[str, ...] = ("acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z")
    thm_cols: Tuple[str, ...] = ("thm_1", "thm_2", "thm_3", "thm_4", "thm_5")
    tof_sensors: int = 5  # 5 ToF sensors with 64 pixels each
    tof_pixels: int = 64  # 8x8 grid per sensor
    early_stopping_patience: int = 2  # Reduce from 4 (faster early stopping)
    grad_clip_norm: float = 1.0 # Keep same
    verbose: int = 1            # Keep same
    kaggle: bool = False
    use_all_sensors: bool = True  # Flag to enable multi-sensor fusion
    quick_test = True

CFG = TestConfig()  # Use full training config instead of test config

# %%
train_data = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_data  = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

# Add this after loading train_data
if CFG.quick_test:
    # Take only first 1000 sequences for quick testing
    train_data = train_data[train_data['sequence_id'].isin(train_data['sequence_id'].unique()[:1000])]
    print(f"Quick test mode: Using {len(train_data['sequence_id'].unique())} sequences")

present_cols = [c for c in CFG.imu_cols if c in train_data.columns]
assert len(present_cols) >= 3, "IMU columns not found as expected."
CFG.imu_cols = tuple(present_cols)

# %%
def interpolate_fill_nan_safe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Interpolate linearly within-sequence, forward/back fill, then
    impute any still-all-NaN columns to 0. Finally, fill any residual NaNs with per-sequence median, then 0.
    """
    out = df.copy()
    # Work only on existing cols (defensive)
    cols = [c for c in cols if c in out.columns]
    out[cols] = out[cols].astype(np.float32)

    # Linear interpolation and ffill/bfill
    out[cols] = out[cols].interpolate(method="linear", limit_direction="both", axis=0)
    out[cols] = out[cols].ffill().bfill()

    # Entire column may still be NaN if sensor never reported in this sequence
    still_all_nan = out[cols].isna().all(axis=0)
    if still_all_nan.any():
        for c in out[cols].columns[still_all_nan]:
            out[c] = 0.0

    # Residual sporadic NaNs -> per-sequence median, then 0
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

def competition_metric(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    idx_to_class: List[str],
    gesture_to_type: Dict[str, str]
) -> Dict[str, float]:
    macro_f1 = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
    y_true_type = np.array([1 if gesture_to_type[idx_to_class[t]] == "Target" else 0 for t in y_true_idx])
    y_pred_type = np.array([1 if gesture_to_type[idx_to_class[p]] == "Target" else 0 for p in y_pred_idx])
    bin_f1 = f1_score(y_true_type, y_pred_type, average="binary", zero_division=0)
    return {"macro_f1": macro_f1, "binary_f1": bin_f1, "mean_f1": (macro_f1 + bin_f1) / 2.0}

# %%
def build_sequence_table(df: pd.DataFrame) -> pd.DataFrame:
    gb = df.groupby("sequence_id", sort=False)
    info = gb.agg(
        subject=("subject", "first"),
        n_rows=("row_id", "count")
    ).reset_index()
    return info

train_seq = build_sequence_table(train_data)
test_seq  = build_sequence_table(test_data)

seq_labels = (
    train_data.groupby("sequence_id", sort=False)
    .agg(gesture=("gesture", "first"), sequence_type=("sequence_type", "first"))
    .reset_index()
)
train_seq = train_seq.merge(seq_labels, on="sequence_id", how="left")

le = LabelEncoder()
train_seq["gesture_idx"] = le.fit_transform(train_seq["gesture"].astype(str))
idx_to_class = list(le.inverse_transform(np.arange(len(le.classes_))))
gesture_to_type = (
    train_seq.drop_duplicates("gesture")[["gesture", "sequence_type"]]
    .set_index("gesture")["sequence_type"].to_dict()
)

# %%
class SequenceDataset(Dataset):
    def __init__(
        self,
        df_raw: pd.DataFrame,
        index_rows: pd.DataFrame,
        imu_cols: Tuple[str, ...],
        max_len: int,
        labels: Optional[np.ndarray] = None,
        use_all_sensors: bool = True,
    ):
        self.df_raw = df_raw
        self.index_rows = index_rows.reset_index(drop=True)
        self.imu_cols = list(imu_cols)
        self.thm_cols = list(CFG.thm_cols) if use_all_sensors else []
        self.use_all_sensors = use_all_sensors
        self.max_len = max_len
        self.labels = labels

    def __len__(self):
        return len(self.index_rows)

    def _prep_one(self, sid: int) -> np.ndarray:
        df = self.df_raw[self.df_raw["sequence_id"] == sid].sort_values("sequence_counter")
        
        # Enhanced feature engineering from notebook (complete set)
        if all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']):
            # Basic features from notebook
            df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
            df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
            df['acc_mag_jerk'] = df['acc_mag'].diff().fillna(0)
            df['rot_angle_vel'] = df['rot_angle'].diff().fillna(0)
            
            # Calculate linear acceleration (gravity removed)
            linear_accel = remove_gravity_from_acc(
                df[['acc_x', 'acc_y', 'acc_z']], 
                df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            )
            df['linear_acc_x'] = linear_accel[:, 0]
            df['linear_acc_y'] = linear_accel[:, 1]
            df['linear_acc_z'] = linear_accel[:, 2]
            df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
            df['linear_acc_mag_jerk'] = df['linear_acc_mag'].diff().fillna(0)
            
            # Calculate angular velocity
            angular_vel = calculate_angular_velocity_from_quat(df[['rot_x', 'rot_y', 'rot_z', 'rot_w']])
            df['angular_vel_x'] = angular_vel[:, 0]
            df['angular_vel_y'] = angular_vel[:, 1]
            df['angular_vel_z'] = angular_vel[:, 2]
            
            # Calculate angular distance
            df['angular_distance'] = calculate_angular_distance(df[['rot_x', 'rot_y', 'rot_z', 'rot_w']])
            
            # Update IMU columns to include all engineered features
            enhanced_imu_cols = list(self.imu_cols) + [
                'acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel',
                'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag', 'linear_acc_mag_jerk',
                'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance'
            ]
            # Filter out columns that don't exist
            enhanced_imu_cols = [c for c in enhanced_imu_cols if c in df.columns]
        else:
            enhanced_imu_cols = self.imu_cols
        
        # Process Enhanced IMU data with notebook-style preprocessing
        df = interpolate_fill_nan_safe(df, enhanced_imu_cols)
        x_imu = df[enhanced_imu_cols].fillna(0).astype(np.float32).values  # (T, C)
        x_imu = resample_to_fixed_length(x_imu, self.max_len)
        # Apply StandardScaler-like normalization per feature (like notebook)
        x_imu = (x_imu - np.mean(x_imu, axis=0, keepdims=True)) / (np.std(x_imu, axis=0, keepdims=True) + 1e-8)
        x_imu = x_imu.T  # (C, L)
        
        if not self.use_all_sensors:
            return x_imu
        
        # Process Thermopile data if available
        all_features = [x_imu]
        
        if self.thm_cols:
            # Check if thermopile data exists and is not all NaN
            has_thm = any(c in df.columns for c in self.thm_cols) and not df[self.thm_cols].isna().all().all()
            if has_thm:
                df = interpolate_fill_nan_safe(df, self.thm_cols)
                x_thm = df[self.thm_cols].astype(np.float32).values
                x_thm = resample_to_fixed_length(x_thm, self.max_len)
                # Apply StandardScaler-like normalization
                x_thm = (x_thm - np.mean(x_thm, axis=0, keepdims=True)) / (np.std(x_thm, axis=0, keepdims=True) + 1e-8)
                x_thm = x_thm.T  # (5, L)
            else:
                # Create zero features if sensors not available
                x_thm = np.zeros((len(self.thm_cols), self.max_len), dtype=np.float32)
            all_features.append(x_thm)
        
        # Enhanced ToF processing with statistical features
        tof_features = []
        for sensor_id in range(1, CFG.tof_sensors + 1):
            tof_cols = [f"tof_{sensor_id}_v{i}" for i in range(CFG.tof_pixels)]
            existing_tof_cols = [c for c in tof_cols if c in df.columns]
            
            if existing_tof_cols and not df[existing_tof_cols].isna().all().all():
                tof_data = df[existing_tof_cols].replace(-1, np.nan)
                
                # Statistical features like in the notebook
                tof_mean = tof_data.mean(axis=1).fillna(0).values
                tof_std = tof_data.std(axis=1).fillna(0).values
                tof_min = tof_data.min(axis=1).fillna(0).values
                tof_max = tof_data.max(axis=1).fillna(0).values
                
                # Combine statistical features
                tof_combined = np.column_stack([tof_mean, tof_std, tof_min, tof_max])  # (T, 4)
                tof_combined = resample_to_fixed_length(tof_combined, self.max_len)
                # Apply StandardScaler-like normalization
                tof_combined = (tof_combined - np.mean(tof_combined, axis=0, keepdims=True)) / (np.std(tof_combined, axis=0, keepdims=True) + 1e-8)
                
                for i in range(4):  # 4 statistical features per sensor
                    tof_features.append(tof_combined[:, i])  # (L,)
            else:
                # Create zero features if sensor not available (4 features per sensor)
                for _ in range(4):
                    tof_features.append(np.zeros(self.max_len, dtype=np.float32))
        
        if tof_features:
            x_tof = np.stack(tof_features)  # (5*4, L) = (20, L)
            all_features.append(x_tof)
        
        # Concatenate all features: IMU + Thermopile + ToF
        x = np.concatenate(all_features, axis=0)
        return x

    # The sequence_id is a string and we will pass 0's for now
    def __getitem__(self, i: int):
        sid = self.index_rows.loc[i, "sequence_id"]
        x = self._prep_one(sid)
        if self.labels is None:
            return torch.tensor(x, dtype=torch.float32), 0
        else:
            y = int(self.labels[i])
            return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long), 0

# %%
# Feature engineering functions from notebook
def remove_gravity_from_acc(acc_data, rot_data):
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data

    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    
    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :] 
            continue

        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError:
             linear_accel[i, :] = acc_values[i, :]
             
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))

    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i+1]

        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
           np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue

        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            pass
            
    return angular_vel
    
def calculate_angular_distance(rot_data):
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)

    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i+1]

        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
           np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0
            continue
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0
            pass
            
    return angular_dist

# MixUp Data Augmentation
class MixupDataset(Dataset):
    def __init__(self, base_dataset, alpha=0.4):
        self.base_dataset = base_dataset
        self.alpha = alpha
        self.indices = np.arange(len(base_dataset))
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        x1, y1, sid1 = self.base_dataset[idx]
        
        # Sample another random index for mixing
        mix_idx = np.random.randint(0, len(self.base_dataset))
        x2, y2, sid2 = self.base_dataset[mix_idx]
        
        # Sample mixing coefficient
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1
        
        # Mix inputs
        mixed_x = lam * x1 + (1 - lam) * x2
        
        # Mix labels (one-hot encoded)
        n_classes = len(le.classes_)
        y1_onehot = torch.zeros(n_classes)
        y2_onehot = torch.zeros(n_classes)
        y1_onehot[y1] = 1.0
        y2_onehot[y2] = 1.0
        mixed_y = lam * y1_onehot + (1 - lam) * y2_onehot
        
        return mixed_x, mixed_y, sid1

# SE Block for attention mechanism
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        se = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # -> (B, C)
        se = F.relu(self.fc1(se))  # -> (B, C//r)
        se = self.sigmoid(self.fc2(se)).unsqueeze(-1)  # -> (B, C, 1)
        return x * se

# Residual CNN Block with SE
class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, drop=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        
        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(drop)
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        out = F.relu(out)
        out = self.pool(out)
        out = self.dropout(out)
        return out

# Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # x: (B, L, H)
        scores = torch.tanh(self.attention(x))  # (B, L, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (B, L)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, H)
        return context

# Two-Branch Model from notebook (converted to PyTorch)
class TwoBranchModel(nn.Module):
    def __init__(self, imu_dim, tof_dim, n_classes, dropout=0.2):
        super().__init__()
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim
        
        # IMU deep branch
        self.imu_branch = nn.Sequential(
            ResidualSECNNBlock(imu_dim, 64, 3, drop=0.1),
            ResidualSECNNBlock(64, 128, 5, drop=0.1)
        )
        
        # TOF/Thermal lighter branch
        self.tof_branch = nn.Sequential(
            nn.Conv1d(tof_dim, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
            nn.Conv1d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2)
        )
        
        # After concatenation, we have 256 channels (128 + 128)
        merged_dim = 256
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(merged_dim, 128, batch_first=True, bidirectional=True)
        # Bidirectional GRU
        self.gru = nn.GRU(merged_dim, 128, batch_first=True, bidirectional=True)
        # Dense layer with noise (approximated with dropout)
        self.dense_noise = nn.Sequential(
            nn.Dropout(0.09),
            nn.Linear(merged_dim, 16),
            nn.ELU()
        )
        
        # After concatenation: 256 (LSTM) + 256 (GRU) + 16 (dense) = 528
        combined_dim = 528
        
        # Attention layer
        self.attention = AttentionLayer(combined_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
        
    def forward(self, x):  # x: (B, C, L)
        # Split input into IMU and TOF parts
        imu = x[:, :self.imu_dim, :]  # (B, imu_dim, L)
        tof = x[:, self.imu_dim:, :]  # (B, tof_dim, L)
        
        # Process through branches
        x1 = self.imu_branch(imu)  # (B, 128, L')
        x2 = self.tof_branch(tof)  # (B, 128, L')
        
        # Concatenate along channel dimension
        merged = torch.cat([x1, x2], dim=1)  # (B, 256, L')
        
        # Transpose for RNN: (B, L', 256)
        merged_t = merged.transpose(1, 2)
        
        # LSTM and GRU processing
        lstm_out, _ = self.lstm(merged_t)  # (B, L', 256)
        gru_out, _ = self.gru(merged_t)    # (B, L', 256)
        
        # Dense processing
        dense_out = self.dense_noise(merged_t)  # (B, L', 16)
        
        # Concatenate all features
        combined = torch.cat([lstm_out, gru_out, dense_out], dim=-1)  # (B, L', 528)
        
        # Apply attention
        attended = self.attention(combined)  # (B, 528)
        
        # Classification
        output = self.classifier(attended)  # (B, n_classes)
        
        return output

# %%
def train_one_epoch(model, loader, optimizer, scheduler, criterion, desc: str):
    model.train()
    loss_sum, n = 0.0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for xb, yb, _ in pbar:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        
        # Handle MixUp labels (soft targets) vs regular labels
        if yb.dim() > 1 and yb.size(1) > 1:  # MixUp soft labels
            # Use soft cross-entropy loss
            loss = -torch.sum(yb * F.log_softmax(logits, dim=1), dim=1).mean()
        else:  # Regular hard labels
            if yb.dim() > 1:
                yb = yb.argmax(dim=1)  # Convert one-hot to indices if needed
            loss = criterion(logits, yb)
            
        loss.backward()
        if CFG.grad_clip_norm:
            nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        bs = xb.size(0)
        loss_sum += loss.item() * bs
        n += bs
        curr = loss_sum / max(1, n)
        # show running average and last LR on the bar
        try:
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
        except Exception:
            lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix(loss=f"{curr:.4f}", lr=f"{lr:.2e}")
    return loss_sum / max(1, n)

@torch.no_grad()
def validate_epoch(model, loader, criterion, desc: str):
    model.eval()
    val_loss_sum, n = 0.0, 0
    all_pred, all_prob, all_sid = [], [], []
    pbar = tqdm(loader, desc=desc, leave=False)
    for xb, yb, sid in pbar:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        bs = xb.size(0)
        val_loss_sum += loss.item() * bs
        n += bs
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = probs.argmax(axis=1)
        all_prob.append(probs)
        all_pred.extend(pred.tolist())
        all_sid.extend(sid.tolist())
        pbar.set_postfix(vloss=f"{val_loss_sum/max(1,n):.4f}")
    all_prob = np.concatenate(all_prob, axis=0) if len(all_prob) else np.zeros((0,0))
    return val_loss_sum / max(1, n), np.array(all_sid), np.array(all_pred), all_prob

# %%
def run_cv_and_train_full():
    set_seed()
    n_classes = len(le.classes_)
    # We'll calculate dimensions per fold since it depends on available features

    y = train_seq["gesture_idx"].values
    subjects = train_seq["subject"].astype(str).values

    splitter = StratifiedGroupKFold(n_splits=CFG.n_folds, shuffle=True, random_state=RANDOM_SEED)
    splits = list(splitter.split(train_seq.index, y, groups=subjects))

    oof_pred = np.full(len(train_seq), -1, dtype=int)
    oof_prob = np.zeros((len(train_seq), n_classes), dtype=np.float32)
    fold_ckpts = []

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        tr_rows = train_seq.loc[tr_idx].reset_index(drop=True)
        va_rows = train_seq.loc[va_idx].reset_index(drop=True)

        tr_ds = SequenceDataset(train_data, tr_rows, CFG.imu_cols, CFG.max_len, tr_rows["gesture_idx"].values, use_all_sensors=CFG.use_all_sensors)
        va_ds = SequenceDataset(train_data, va_rows, CFG.imu_cols, CFG.max_len, va_rows["gesture_idx"].values, use_all_sensors=CFG.use_all_sensors)

        # Add MixUp augmentation to training dataset
        tr_ds_mixup = MixupDataset(tr_ds, alpha=0.4)  # Same alpha as notebook
        
        tr_loader = DataLoader(tr_ds_mixup, batch_size=CFG.batch_size, shuffle=True,  num_workers=CFG.num_workers, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

        # Calculate dimensions for two-branch model
        if CFG.use_all_sensors:
            # Enhanced IMU features: original (7) + basic (4) + advanced (9) = 20 total
            imu_dim = len(CFG.imu_cols) + 13  # 7 original + 13 engineered features
            tof_dim = len(CFG.thm_cols) + CFG.tof_sensors * 4  # thermopile + 4 stats per ToF sensor
        else:
            imu_dim = len(CFG.imu_cols) + 13  # original + engineered
            tof_dim = 0
        
        model = TwoBranchModel(imu_dim=imu_dim, tof_dim=tof_dim, n_classes=n_classes, dropout=CFG.dropout).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        steps_per_epoch = max(1, len(tr_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.lr, epochs=CFG.n_epochs, steps_per_epoch=steps_per_epoch)
        criterion = nn.CrossEntropyLoss()

        best_score = -1.0
        best_path = os.path.join(OUTPUT_DIR, f"cnn1d_fold{fold}.pt")
        patience = CFG.early_stopping_patience
        epochs_no_improve = 0

        for epoch in range(1, CFG.n_epochs + 1):
            tr_desc = f"Fold {fold} | Epoch {epoch}/{CFG.n_epochs} | train"
            tr_loss = train_one_epoch(model, tr_loader, optimizer, scheduler, criterion, tr_desc)

            va_desc = f"Fold {fold} | Epoch {epoch}/{CFG.n_epochs} | valid"
            va_loss, sids, preds, probs = validate_epoch(model, va_loader, criterion, va_desc)

            # Metrics
            y_true = va_rows["gesture_idx"].values
            metrics = competition_metric(y_true, preds, idx_to_class, gesture_to_type)

            tqdm.write(
                f"Fold {fold} | Epoch {epoch:02d} "
                f"| tr_loss {tr_loss:.4f} | va_loss {va_loss:.4f} "
                f"| macroF1 {metrics['macro_f1']:.4f} | binF1 {metrics['binary_f1']:.4f} | meanF1 {metrics['mean_f1']:.4f}"
            )

            # Track best by meanF1
            if metrics["mean_f1"] > best_score:
                best_score = metrics["mean_f1"]
                torch.save(model.state_dict(), best_path)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    tqdm.write(f"  Early stopping on fold {fold} at epoch {epoch}")
                    break

        # Load best & produce OOF for this fold
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        va_loader_eval = DataLoader(va_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
        _, sids, preds, probs = validate_epoch(model, va_loader_eval, criterion, f"Fold {fold} | OOF")
        # Map into oof arrays
        oof_pred[va_idx] = preds
        oof_prob[va_idx] = probs
        fold_ckpts.append(best_path)

    # CV metrics
    metrics = competition_metric(
        train_seq["gesture_idx"].values,
        oof_pred,
        idx_to_class,
        gesture_to_type
    )
    print(f"\nCV macroF1={metrics['macro_f1']:.4f} | binF1={metrics['binary_f1']:.4f} | meanF1={metrics['mean_f1']:.4f}")

    # Save meta for inference
    cfg_dict = asdict(CFG)
    # Ensure all sensor configs are saved
    cfg_dict["use_all_sensors"] = CFG.use_all_sensors
    cfg_dict["thm_cols"] = list(CFG.thm_cols)
    cfg_dict["tof_sensors"] = CFG.tof_sensors
    cfg_dict["tof_pixels"] = CFG.tof_pixels
    cfg_dict["imu_dim"] = imu_dim
    cfg_dict["tof_dim"] = tof_dim
    meta = {
        "cfg": cfg_dict,
        "fold_ckpts": fold_ckpts,
        "classes": idx_to_class,
        "gesture_to_type": gesture_to_type
    }
    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return fold_ckpts

# %%
fold_ckpts = run_cv_and_train_full()

# %%
class InferenceEnsemble:
    def __init__(self, ckpt_paths: List[str], imu_dim: int, tof_dim: int, n_classes: int, device=DEVICE, dropout=0.0):
        self.device = device
        self.models = []
        for p in ckpt_paths:
            m = TwoBranchModel(imu_dim=imu_dim, tof_dim=tof_dim, n_classes=n_classes, dropout=dropout).to(device)
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
        # Enhanced feature engineering for inference (match training)
        if all(col in seq_df.columns for col in ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']):
            # Basic features from notebook
            seq_df['acc_mag'] = np.sqrt(seq_df['acc_x']**2 + seq_df['acc_y']**2 + seq_df['acc_z']**2)
            seq_df['rot_angle'] = 2 * np.arccos(seq_df['rot_w'].clip(-1, 1))
            seq_df['acc_mag_jerk'] = seq_df['acc_mag'].diff().fillna(0)
            seq_df['rot_angle_vel'] = seq_df['rot_angle'].diff().fillna(0)
            
            # Advanced features
            linear_accel = remove_gravity_from_acc(
                seq_df[['acc_x', 'acc_y', 'acc_z']], 
                seq_df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            )
            seq_df['linear_acc_x'] = linear_accel[:, 0]
            seq_df['linear_acc_y'] = linear_accel[:, 1]
            seq_df['linear_acc_z'] = linear_accel[:, 2]
            seq_df['linear_acc_mag'] = np.sqrt(seq_df['linear_acc_x']**2 + seq_df['linear_acc_y']**2 + seq_df['linear_acc_z']**2)
            seq_df['linear_acc_mag_jerk'] = seq_df['linear_acc_mag'].diff().fillna(0)
            
            angular_vel = calculate_angular_velocity_from_quat(seq_df[['rot_x', 'rot_y', 'rot_z', 'rot_w']])
            seq_df['angular_vel_x'] = angular_vel[:, 0]
            seq_df['angular_vel_y'] = angular_vel[:, 1]
            seq_df['angular_vel_z'] = angular_vel[:, 2]
            
            seq_df['angular_distance'] = calculate_angular_distance(seq_df[['rot_x', 'rot_y', 'rot_z', 'rot_w']])
            
            enhanced_imu_cols = list(CFG.imu_cols) + [
                'acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel',
                'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag', 'linear_acc_mag_jerk',
                'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance'
            ]
            enhanced_imu_cols = [c for c in enhanced_imu_cols if c in seq_df.columns]
        else:
            enhanced_imu_cols = list(CFG.imu_cols)
        
        # Process Enhanced IMU data
        seq_df = interpolate_fill_nan_safe(seq_df, enhanced_imu_cols)
        x_imu = seq_df[enhanced_imu_cols].fillna(0).astype(np.float32).values
        x_imu = resample_to_fixed_length(x_imu, CFG.max_len)
        # Apply StandardScaler-like normalization
        x_imu = (x_imu - np.mean(x_imu, axis=0, keepdims=True)) / (np.std(x_imu, axis=0, keepdims=True) + 1e-8)
        x_imu = x_imu.T
        
        all_features = [x_imu]
        
        if CFG.use_all_sensors:
            # Process Thermopile data
            thm_cols = list(CFG.thm_cols)
            has_thm = any(c in seq_df.columns for c in thm_cols) and not seq_df[thm_cols].isna().all().all()
            if has_thm:
                seq_df = interpolate_fill_nan_safe(seq_df, thm_cols)
                x_thm = seq_df[thm_cols].astype(np.float32).values
                x_thm = resample_to_fixed_length(x_thm, CFG.max_len)
                x_thm = zscore_per_sequence(x_thm).T
            else:
                x_thm = np.zeros((len(thm_cols), CFG.max_len), dtype=np.float32)
            all_features.append(x_thm)
            
            # Enhanced ToF processing with statistical features
            tof_features = []
            for sensor_id in range(1, CFG.tof_sensors + 1):
                tof_cols = [f"tof_{sensor_id}_v{i}" for i in range(CFG.tof_pixels)]
                existing_tof_cols = [c for c in tof_cols if c in seq_df.columns]
                
                if existing_tof_cols and not seq_df[existing_tof_cols].isna().all().all():
                    tof_data = seq_df[existing_tof_cols].replace(-1, np.nan)
                    
                    # Statistical features
                    tof_mean = tof_data.mean(axis=1).fillna(0).values
                    tof_std = tof_data.std(axis=1).fillna(0).values
                    tof_min = tof_data.min(axis=1).fillna(0).values
                    tof_max = tof_data.max(axis=1).fillna(0).values
                    
                    tof_combined = np.column_stack([tof_mean, tof_std, tof_min, tof_max])
                    tof_combined = resample_to_fixed_length(tof_combined, CFG.max_len)
                    tof_combined = zscore_per_sequence(tof_combined)
                    
                    for i in range(4):
                        tof_features.append(tof_combined[:, i])
                else:
                    for _ in range(4):
                        tof_features.append(np.zeros(CFG.max_len, dtype=np.float32))
            
            if tof_features:
                x_tof = np.stack(tof_features)
                all_features.append(x_tof)
        
        # Concatenate all features
        x = np.concatenate(all_features, axis=0)
        xb = torch.tensor(x[None, ...], dtype=torch.float32, device=self.device)
        probs = self.predict_proba_batch(xb)[0]
        pred_idx = int(np.argmax(probs))
        return idx_to_class[pred_idx]

# %%
@torch.no_grad()
def predict_test_and_save_csv():
    meta = json.load(open(os.path.join(OUTPUT_DIR, "meta.json")))
    ckpts = meta["fold_ckpts"]
    # Use correct dimensions for two-branch model
    if CFG.use_all_sensors:
        imu_dim = len(CFG.imu_cols) + 13  # 7 original + 13 engineered features
        tof_dim = len(CFG.thm_cols) + CFG.tof_sensors * 4  # thermopile + ToF stats
    else:
        imu_dim = len(CFG.imu_cols) + 13  # original + engineered
        tof_dim = 0
    ens = InferenceEnsemble(ckpts, imu_dim, tof_dim, len(idx_to_class), device=DEVICE, dropout=0.0)

    test_rows = test_seq.copy().reset_index(drop=True)
    preds = []
    for _, row in tqdm(test_rows.iterrows(), total=len(test_rows), desc="Predict test"):
        sid = row["sequence_id"]
        seq_df = test_data[test_data["sequence_id"] == sid].sort_values("sequence_counter")
        label = ens.predict_label_for_sequence_df(seq_df)
        preds.append({"sequence_id": sid, "gesture": label})

    sub = pd.DataFrame(preds)
    sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to: {sub_path}")

predict_test_and_save_csv()

# %%
if CFG.kaggle == True:
    import kaggle_evaluation.cmi_inference_server

    _ENSEMBLE = None
    def _get_ensemble():
        global _ENSEMBLE
        if _ENSEMBLE is None:
            meta = json.load(open(os.path.join(OUTPUT_DIR, "meta.json")))
            ckpts = meta["fold_ckpts"]
            # Use correct dimensions for two-branch model
            if CFG.use_all_sensors:
                imu_dim = len(CFG.imu_cols) + 13  # 7 original + 13 engineered features
                tof_dim = len(CFG.thm_cols) + CFG.tof_sensors * 4  # thermopile + ToF stats
            else:
                imu_dim = len(CFG.imu_cols) + 13  # original + engineered
                tof_dim = 0
            _ENSEMBLE = InferenceEnsemble(ckpts, imu_dim, tof_dim, len(idx_to_class), device=DEVICE, dropout=0.0)
        return _ENSEMBLE

    def predict(sequence_df: pl.DataFrame, demographics: pl.DataFrame) -> str:
        """
        Kaggle will call this repeatedly, one sequence at a time.
        Input: DataFrame with at least IMU columns (acc_*, rot_* if available).
        Returns: gesture label as string.
        """
        # Ensure missing IMU columns are present for pipeline consistency
        for c in CFG.imu_cols:
            if c not in sequence_df.columns:
                sequence_df[c] = np.nan
        ens = _get_ensemble()
        return ens.predict_label_for_sequence_df(sequence_df)

    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            data_paths=(
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
            )
        )


# %%
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Script ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
print(f"Total execution time: {execution_time_minutes:.2f} minutes")