# CV macroF1=0.4893 | binF1=0.9219 | meanF1=0.7056

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
from torch.utils.data import Dataset, DataLoader

# %%  
INPUT_DIR = "./kaggle_cmi_2025/data"
OUTPUT_DIR = "./multi_data_result"
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

print(DEVICE)
# %%
import time
start_time = time.time()
print(f"Script started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")


# %%
@dataclass
class Config:
    max_len: int = 400
    batch_size: int = 32
    num_workers: int = 8
    n_folds: int = 3
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
CFG = Config()

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
        
        # Process IMU data (always present)
        df = interpolate_fill_nan_safe(df, self.imu_cols)
        x_imu = df[self.imu_cols].astype(np.float32).values  # (T, C)
        x_imu = resample_to_fixed_length(x_imu, self.max_len)
        x_imu = zscore_per_sequence(x_imu)
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
                x_thm = zscore_per_sequence(x_thm)
                x_thm = x_thm.T  # (5, L)
            else:
                # Create zero features if sensors not available
                x_thm = np.zeros((len(self.thm_cols), self.max_len), dtype=np.float32)
            all_features.append(x_thm)
        
        # Process ToF data (simplified: average across all pixels for each sensor)
        tof_features = []
        for sensor_id in range(1, CFG.tof_sensors + 1):
            tof_cols = [f"tof_{sensor_id}_v{i}" for i in range(CFG.tof_pixels)]
            # Check if these columns exist
            existing_tof_cols = [c for c in tof_cols if c in df.columns]
            
            if existing_tof_cols and not df[existing_tof_cols].isna().all().all():
                # Average across all pixels for this sensor, handle -1 values
                tof_data = df[existing_tof_cols].replace(-1, np.nan).mean(axis=1).fillna(0).values
                tof_data = tof_data.reshape(-1, 1)  # (T, 1)
                tof_data = resample_to_fixed_length(tof_data, self.max_len)
                tof_data = zscore_per_sequence(tof_data)
                tof_features.append(tof_data.T[0])  # (L,)
            else:
                tof_features.append(np.zeros(self.max_len, dtype=np.float32))
        
        if tof_features:
            x_tof = np.stack(tof_features)  # (5, L)
            all_features.append(x_tof)
        
        # Concatenate all features
        x = np.concatenate(all_features, axis=0)  # (7+5+5, L) = (17, L)
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

    def forward(self, x):  # x: (B, C, L)
        x = self.net(x)
        x = self.gap(x).squeeze(-1)
        return self.head(x)

# %%
def train_one_epoch(model, loader, optimizer, scheduler, criterion, desc: str):
    model.train()
    loss_sum, n = 0.0, 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for xb, yb, _ in pbar:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
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
    # Calculate total input channels based on sensor configuration
    if CFG.use_all_sensors:
        in_ch = len(CFG.imu_cols) + len(CFG.thm_cols) + CFG.tof_sensors  # 7 + 5 + 5 = 17
    else:
        in_ch = len(CFG.imu_cols)  # 7

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

        tr_loader = DataLoader(tr_ds, batch_size=CFG.batch_size, shuffle=True,  num_workers=CFG.num_workers, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

        model = CNN1D(in_ch=in_ch, n_classes=n_classes, dropout=CFG.dropout).to(DEVICE)
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
# fold_ckpts = run_cv_and_train_full()

# %%
class InferenceEnsemble:
    def __init__(self, ckpt_paths: List[str], in_ch: int, n_classes: int, device=DEVICE, dropout=0.0):
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

    def predict_label_for_sequence_df(self, seq_df: pd.DataFrame) -> str:
        # Prepare single sequence with all sensors
        # Process IMU data
        seq_df = interpolate_fill_nan_safe(seq_df, list(CFG.imu_cols))
        x_imu = seq_df[list(CFG.imu_cols)].astype(np.float32).values
        x_imu = resample_to_fixed_length(x_imu, CFG.max_len)
        x_imu = zscore_per_sequence(x_imu).T  # (7, L)
        
        all_features = [x_imu]
        
        if CFG.use_all_sensors:
            # Process Thermopile data
            thm_cols = list(CFG.thm_cols)
            has_thm = any(c in seq_df.columns for c in thm_cols) and not seq_df[thm_cols].isna().all().all()
            if has_thm:
                seq_df = interpolate_fill_nan_safe(seq_df, thm_cols)
                x_thm = seq_df[thm_cols].astype(np.float32).values
                x_thm = resample_to_fixed_length(x_thm, CFG.max_len)
                x_thm = zscore_per_sequence(x_thm).T  # (5, L)
            else:
                x_thm = np.zeros((len(thm_cols), CFG.max_len), dtype=np.float32)
            all_features.append(x_thm)
            
            # Process ToF data (simplified: average across pixels)
            tof_features = []
            for sensor_id in range(1, CFG.tof_sensors + 1):
                tof_cols = [f"tof_{sensor_id}_v{i}" for i in range(CFG.tof_pixels)]
                existing_tof_cols = [c for c in tof_cols if c in seq_df.columns]
                
                if existing_tof_cols and not seq_df[existing_tof_cols].isna().all().all():
                    tof_data = seq_df[existing_tof_cols].replace(-1, np.nan).mean(axis=1).fillna(0).values
                    tof_data = tof_data.reshape(-1, 1)
                    tof_data = resample_to_fixed_length(tof_data, CFG.max_len)
                    tof_data = zscore_per_sequence(tof_data)
                    tof_features.append(tof_data.T[0])
                else:
                    tof_features.append(np.zeros(CFG.max_len, dtype=np.float32))
            
            if tof_features:
                x_tof = np.stack(tof_features)  # (5, L)
                all_features.append(x_tof)
        
        # Concatenate all features
        x = np.concatenate(all_features, axis=0)  # (17, L) if all sensors, (7, L) if IMU only
        xb = torch.tensor(x[None, ...], dtype=torch.float32, device=self.device)
        probs = self.predict_proba_batch(xb)[0]
        pred_idx = int(np.argmax(probs))
        return idx_to_class[pred_idx]

# %%
@torch.no_grad()
def predict_test_and_save_csv():
    meta = json.load(open(os.path.join(OUTPUT_DIR, "meta.json")))
    ckpts = meta["fold_ckpts"]
    # Use correct number of input channels
    if CFG.use_all_sensors:
        in_ch = len(CFG.imu_cols) + len(CFG.thm_cols) + CFG.tof_sensors  # 17
    else:
        in_ch = len(CFG.imu_cols)  # 7
    ens = InferenceEnsemble(ckpts, in_ch, len(idx_to_class), device=DEVICE, dropout=0.0)

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

# predict_test_and_save_csv()

# %%
if CFG.kaggle == True:
    import kaggle_evaluation.cmi_inference_server

    _ENSEMBLE = None
    def _get_ensemble():
        global _ENSEMBLE
        if _ENSEMBLE is None:
            meta = json.load(open(os.path.join(OUTPUT_DIR, "meta.json")))
            ckpts = meta["fold_ckpts"]
            # Use correct number of input channels
            if CFG.use_all_sensors:
                in_ch = len(CFG.imu_cols) + len(CFG.thm_cols) + CFG.tof_sensors  # 17
            else:
                in_ch = len(CFG.imu_cols)  # 7
            _ENSEMBLE = InferenceEnsemble(ckpts, in_ch, len(idx_to_class), device=DEVICE, dropout=0.0)
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


# %% main
if __name__ == "__main__":
    fold_ckpts = run_cv_and_train_full()
    predict_test_and_save_csv()