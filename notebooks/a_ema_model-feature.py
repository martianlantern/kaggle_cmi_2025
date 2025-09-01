# full_end_to_end_cmi.py
# End-to-end training + inference using CMIModel (ResNetSE per-modality + BERT head)
# Keeps your Config/TestConfig and training/inference scaffolding.

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
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation as R

# ========== Configs (preserved) ==========
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

# Set config you want to use
CFG = Config()
# CFG = TestConfig()  # for quick local tests

# ========== Environment & seed ==========
INPUT_DIR = "../kaggle_cmi_2025/data"
OUTPUT_DIR = "./notebook1_result"
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
print("Device:", DEVICE)

# ========== Utilities (interpolation/resample/zscore) ==========
def interpolate_fill_nan_safe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in cols if c in out.columns]
    if not cols:
        return out
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
    T, C = arr.shape
    if T == out_len:
        return arr.astype(np.float32)
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
    m = np.nanmean(x, axis=0)
    s = np.nanstd(x, axis=0)
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

# ========== Sequence table and label encoder ==========
train_data = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_data  = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

if CFG.quick_test:
    train_data = train_data[train_data['sequence_id'].isin(train_data['sequence_id'].unique()[:1000])]
    print(f"Quick test mode: Using {len(train_data['sequence_id'].unique())} sequences")

present_cols = [c for c in CFG.imu_cols if c in train_data.columns]
assert len(present_cols) >= 3, "IMU columns not found as expected."
CFG.imu_cols = tuple(present_cols)

def build_sequence_table(df: pd.DataFrame) -> pd.DataFrame:
    gb = df.groupby("sequence_id", sort=False)
    info = gb.agg(subject=("subject", "first"), n_rows=("row_id", "count")).reset_index()
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

# ========== SequenceDataset (keeps your interpolation/resampling logic) ==========
class SequenceDataset(Dataset):
    """
    Builds per-sequence multi-sensor features and returns (imu, thm, tof, label)
    imu: (pad_len, imu_channels)
    thm: (pad_len, thm_channels) possibly zeros
    tof: (pad_len, tof_channels) simplified (one channel per sensor)
    """
    def __init__(self, df_raw: pd.DataFrame, index_rows: pd.DataFrame, imu_cols: Tuple[str, ...],
                 max_len: int, labels: Optional[np.ndarray] = None, use_all_sensors: bool = True):
        self.df_raw = df_raw
        self.index_rows = index_rows.reset_index(drop=True)
        self.imu_cols = list(imu_cols)
        self.thm_cols = list(CFG.thm_cols) if use_all_sensors else []
        self.use_all_sensors = use_all_sensors
        self.max_len = max_len
        self.labels = labels

    def __len__(self):
        return len(self.index_rows)

    def _prep_one(self, sid: int):
        df = self.df_raw[self.df_raw["sequence_id"] == sid].sort_values("sequence_counter")
        # IMU
        df_imu = interpolate_fill_nan_safe(df, self.imu_cols)
        x_imu = df_imu[self.imu_cols].astype(np.float32).values  # (T, C_imu)
        x_imu = resample_to_fixed_length(x_imu, self.max_len)
        x_imu = zscore_per_sequence(x_imu).astype(np.float32)   # (L, C)
        # keep as (L, C)

        # Thermopile
        if self.use_all_sensors and len(self.thm_cols) > 0:
            has_thm = any(c in df.columns for c in self.thm_cols) and not df[self.thm_cols].isna().all().all()
            if has_thm:
                df_thm = interpolate_fill_nan_safe(df, self.thm_cols)
                x_thm = df_thm[self.thm_cols].astype(np.float32).values
                x_thm = resample_to_fixed_length(x_thm, self.max_len)
                x_thm = zscore_per_sequence(x_thm).astype(np.float32)
            else:
                x_thm = np.zeros((self.max_len, len(self.thm_cols)), dtype=np.float32)
        else:
            x_thm = np.zeros((self.max_len, 0), dtype=np.float32)

        # ToF simplified: average pixels per sensor -> single channel per sensor
        if self.use_all_sensors:
            tof_feats = []
            for sensor_id in range(1, CFG.tof_sensors+1):
                tof_cols = [f"tof_{sensor_id}_v{i}" for i in range(CFG.tof_pixels)]
                existing = [c for c in tof_cols if c in df.columns]
                if existing and not df[existing].replace(-1, np.nan).isna().all().all():
                    tof_data = df[existing].replace(-1, np.nan).mean(axis=1).fillna(0).values.reshape(-1,1)
                    tof_data = resample_to_fixed_length(tof_data, self.max_len)
                    tof_data = zscore_per_sequence(tof_data).astype(np.float32)
                    tof_feats.append(tof_data[:,0])
                else:
                    tof_feats.append(np.zeros(self.max_len, dtype=np.float32))
            x_tof = np.stack(tof_feats, axis=1)  # (L, sensors)
        else:
            x_tof = np.zeros((self.max_len, 0), dtype=np.float32)

        return x_imu, x_thm, x_tof

    def __getitem__(self, i):
        sid = self.index_rows.loc[i, "sequence_id"]
        x_imu, x_thm, x_tof = self._prep_one(sid)
        # Return as (L, C) — training loop will move to device and model expects (B,L,C)
        if self.labels is None:
            return torch.tensor(x_imu, dtype=torch.float32), torch.tensor(x_thm, dtype=torch.float32), torch.tensor(x_tof, dtype=torch.float32), 0
        else:
            y = int(self.labels[i])
            return torch.tensor(x_imu, dtype=torch.float32), torch.tensor(x_thm, dtype=torch.float32), torch.tensor(x_tof, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

# ========== Model: SE, ResNetSEBlock, CMIModel (core model replacement) ==========
class SEBlock(nn.Module):
    def __init__(self, channels, reduction = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # x: (B, C, L)
        se = F.adaptive_avg_pool1d(x, 1).squeeze(-1)      # -> (B, C)
        se = F.relu(self.fc1(se), inplace=True)          # -> (B, C//r)
        se = self.sigmoid(self.fc2(se)).unsqueeze(-1)    # -> (B, C, 1)
        return x * se

class ResNetSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, wd = 1e-4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        return self.relu(out)

class CMIModel(nn.Module):
    def __init__(self, imu_dim, thm_dim, tof_dim, n_classes, **kwargs):
        super().__init__()
        # config with safe defaults
        feat_dim = kwargs.get("feat_dim", 256)
        imu1_channels = kwargs.get("imu1_channels", 128)
        imu1_layers = kwargs.get("imu1_layers", 1)
        imu1_dropout = kwargs.get("imu1_dropout", 0.3)
        imu2_layers = kwargs.get("imu2_layers", 1)
        imu2_dropout = kwargs.get("imu2_dropout", 0.3)

        thm1_channels = kwargs.get("thm1_channels", 64)
        thm1_dropout = kwargs.get("thm1_dropout", 0.2)
        thm2_dropout = kwargs.get("thm2_dropout", 0.2)

        tof1_channels = kwargs.get("tof1_channels", 64)
        tof1_dropout = kwargs.get("tof1_dropout", 0.2)
        tof2_dropout = kwargs.get("tof2_dropout", 0.2)

        bert_layers = kwargs.get("bert_layers", 4)
        bert_heads = kwargs.get("bert_heads", 8)

        cls1_channels = kwargs.get("cls1_channels", 512)
        cls2_channels = kwargs.get("cls2_channels", 128)
        cls1_dropout = kwargs.get("cls1_dropout", 0.2)
        cls2_dropout = kwargs.get("cls2_dropout", 0.2)

        # IMU branch
        imu_blocks = []
        in_ch = imu_dim
        for i in range(max(1, imu1_layers)):
            out_ch = imu1_channels
            imu_blocks.append(ResNetSEBlock(in_ch, out_ch))
            in_ch = out_ch
        imu_blocks.append(ResNetSEBlock(in_ch, feat_dim))
        imu_blocks.append(nn.MaxPool1d(2, ceil_mode=True))
        imu_blocks.append(nn.Dropout(imu2_dropout))
        self.imu_branch = nn.Sequential(*imu_blocks)

        # THM branch
        self.thm_branch = nn.Sequential(
            nn.Conv1d(thm_dim, thm1_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(thm1_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, ceil_mode=True),
            nn.Dropout(thm1_dropout),

            nn.Conv1d(thm1_channels, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, ceil_mode=True),
            nn.Dropout(thm2_dropout)
        )

        # ToF branch
        self.tof_branch = nn.Sequential(
            nn.Conv1d(tof_dim, tof1_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(tof1_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, ceil_mode=True),
            nn.Dropout(tof1_dropout),

            nn.Conv1d(tof1_channels, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, ceil_mode=True),
            nn.Dropout(tof2_dropout)
        )

        # CLS token + BERT head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, feat_dim))
        from transformers import BertConfig, BertModel
        bert_cfg = BertConfig(
            hidden_size=feat_dim,
            num_hidden_layers=bert_layers,
            num_attention_heads=bert_heads,
            intermediate_size=feat_dim * 4
        )
        self.bert = BertModel(bert_cfg)

        # classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, cls1_channels, bias=False),
            nn.BatchNorm1d(cls1_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(cls1_dropout),
            nn.Linear(cls1_channels, cls2_channels, bias=False),
            nn.BatchNorm1d(cls2_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(cls2_dropout),
            nn.Linear(cls2_channels, n_classes)
        )

    def forward(self, imu, thm, tof):
        # inputs: (B, L, C) each. Permute to (B, C, L) for conv1d
        B = imu.size(0)
        device = imu.device
        # handle zero-dim branches gracefully
        if imu.numel() == 0:
            imu_feat = torch.zeros((B, self.bert.config.hidden_size, 1), device=device)
        else:
            imu_feat = self.imu_branch(imu.permute(0,2,1))

        if thm.numel() == 0:
            thm_feat = torch.zeros_like(imu_feat)
        else:
            thm_feat = self.thm_branch(thm.permute(0,2,1))

        if tof.numel() == 0:
            tof_feat = torch.zeros_like(imu_feat)
        else:
            tof_feat = self.tof_branch(tof.permute(0,2,1))

        # to time-major (B, T, H)
        def to_time(x):
            return x.permute(0,2,1)
        seqs = []
        for f in (imu_feat, thm_feat, tof_feat):
            if f is None or f.numel() == 0:
                continue
            seqs.append(to_time(f))
        if len(seqs) == 0:
            batch = torch.zeros((B,1,self.bert.config.hidden_size), device=device)
        else:
            max_T = max(s.size(1) for s in seqs)
            padded = []
            for s in (imu_feat, thm_feat, tof_feat):
                if s is None or s.numel() == 0:
                    padded.append(torch.zeros((B, max_T, self.bert.config.hidden_size), device=device))
                else:
                    t = to_time(s)
                    if t.size(1) < max_T:
                        pad = torch.zeros((B, max_T - t.size(1), t.size(2)), device=device)
                        padded.append(torch.cat([t, pad], dim=1))
                    else:
                        padded.append(t[:, :max_T, :])
            batch = padded[0] + padded[1] + padded[2]

        cls = self.cls_token.expand(B, -1, -1)
        bert_input = torch.cat([cls, batch], dim=1)
        outputs = self.bert(inputs_embeds=bert_input)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_emb)

# ========== EMA helper ==========
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.model = model
        self.shadow = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().cpu().clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new = (1.0 - self.decay) * param.detach().cpu() + self.decay * self.shadow[name]
                self.shadow[name] = new.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].to(param.device))

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state):
        self.shadow = {k: v.clone() for k,v in state.items()}

# ========== Cross-validation flow using SequenceDataset and CMIModel ==========
def run_cv_and_train_full():
    set_seed()
    n_classes = len(le.classes_)
    y = train_seq["gesture_idx"].values
    subjects = train_seq["subject"].astype(str).values

    splitter = StratifiedGroupKFold(n_splits=CFG.n_folds, shuffle=True, random_state=RANDOM_SEED)
    splits = list(splitter.split(train_seq.index, y, groups=subjects))

    oof_pred = np.full(len(train_seq), -1, dtype=int)
    oof_prob = np.zeros((len(train_seq), n_classes), dtype=np.float32)
    fold_ckpts = []

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        print(f"\n=== Fold {fold}/{CFG.n_folds} ===")
        tr_rows = train_seq.loc[tr_idx].reset_index(drop=True)
        va_rows = train_seq.loc[va_idx].reset_index(drop=True)

        tr_ds = SequenceDataset(train_data, tr_rows, CFG.imu_cols, CFG.max_len, tr_rows["gesture_idx"].values, use_all_sensors=CFG.use_all_sensors)
        va_ds = SequenceDataset(train_data, va_rows, CFG.imu_cols, CFG.max_len, va_rows["gesture_idx"].values, use_all_sensors=CFG.use_all_sensors)

        tr_loader = DataLoader(tr_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=True)

        # deduce dims for model
        # SequenceDataset returns tensors shaped (L, C) so take first item to infer dims
        sample_imu, sample_thm, sample_tof, _ = tr_ds[0]
        imu_dim = sample_imu.shape[1]
        thm_dim = sample_thm.shape[1] if sample_thm.numel() > 0 else 0
        tof_dim = sample_tof.shape[1] if sample_tof.numel() > 0 else 0

        model = CMIModel(imu_dim, thm_dim, tof_dim, n_classes,
                         feat_dim=500,
                         imu1_channels=219, imu1_layers=0, imu1_dropout=0.2946731587132302, imu2_layers=0, imu2_dropout=0.2697745571929592,
                         thm1_channels=82, thm1_dropout=0.2641274454844602, thm2_dropout=0.302896343020985,
                         tof1_channels=82, tof1_dropout=0.2641274454844602, tof2_dropout=0.3028963430209852,
                         bert_layers=8, bert_heads=10,
                         cls1_channels=937, cls2_channels=303, cls1_dropout=0.2281834512100508, cls2_dropout=0.22502521933558461
                         ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        steps_per_epoch = max(1, len(tr_loader))
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=CFG.lr, epochs=CFG.n_epochs, steps_per_epoch=steps_per_epoch)
        criterion = nn.CrossEntropyLoss()

        ema = EMA(model, decay=0.999)
        best_score = -1.0
        best_path = os.path.join(OUTPUT_DIR, f"cmi_fold{fold}.pt")
        patience = CFG.early_stopping_patience
        epochs_no_improve = 0

        for epoch in range(1, CFG.n_epochs + 1):
            # --- train epoch ---
            model.train()
            loss_sum, n = 0.0, 0
            pbar = tqdm(tr_loader, desc=f"Fold {fold} Epoch {epoch} train", leave=False)
            for imu, thm, tof, y in pbar:
                # inputs provided as (L, C) per sample -> DataLoader stacks to (B, L, C)
                imu = imu.to(DEVICE)
                thm = thm.to(DEVICE) if thm.numel() else torch.zeros((imu.size(0), imu.size(1), 0), device=DEVICE)
                tof = tof.to(DEVICE) if tof.numel() else torch.zeros((imu.size(0), imu.size(1), 0), device=DEVICE)
                y = y.to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                logits = model(imu, thm, tof)
                loss = criterion(logits, y)
                loss.backward()
                if CFG.grad_clip_norm:
                    nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip_norm)
                optimizer.step()
                scheduler.step()
                ema.update()
                bs = imu.size(0)
                loss_sum += loss.item() * bs
                n += bs
                pbar.set_postfix(loss=f"{loss_sum / max(1, n):.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            # --- validate epoch ---
            model.eval()
            val_loss_sum, n = 0.0, 0
            all_preds, all_probs, all_sids = [], [], []
            pbar = tqdm(va_loader, desc=f"Fold {fold} Epoch {epoch} valid", leave=False)
            with torch.no_grad():
                for imu, thm, tof, y in pbar:
                    imu = imu.to(DEVICE)
                    thm = thm.to(DEVICE) if thm.numel() else torch.zeros((imu.size(0), imu.size(1), 0), device=DEVICE)
                    tof = tof.to(DEVICE) if tof.numel() else torch.zeros((imu.size(0), imu.size(1), 0), device=DEVICE)
                    y = y.to(DEVICE)
                    logits = model(imu, thm, tof)
                    loss = criterion(logits, y)
                    val_loss_sum += loss.item() * imu.size(0)
                    n += imu.size(0)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()
                    preds = probs.argmax(axis=1)
                    all_probs.append(probs)
                    all_preds.extend(preds.tolist())
                    all_sids.extend([0]*len(preds))
                    pbar.set_postfix(vloss=f"{val_loss_sum/max(1,n):.4f}")

            all_probs = np.concatenate(all_probs, axis=0) if len(all_probs) else np.zeros((0,0))
            y_true = va_rows["gesture_idx"].values
            metrics = competition_metric(y_true, np.array(all_preds), idx_to_class, gesture_to_type)

            tqdm.write(
                f"Fold {fold} | Epoch {epoch:02d}"
                f" | tr_loss {loss_sum/n:.4f}" if n>0 else ""
                f" | va_loss {val_loss_sum/max(1,n):.4f} | macroF1 {metrics['macro_f1']:.4f} | binF1 {metrics['binary_f1']:.4f} | meanF1 {metrics['mean_f1']:.4f}"
            )

            if metrics["mean_f1"] > best_score:
                best_score = metrics["mean_f1"]
                # save EMA weights (apply shadow to model copy)
                ema.apply_shadow()
                torch.save(model.state_dict(), best_path)
                # restore training weights not strictly needed; we'll reload model later from path
                epochs_no_improve = 0
                tqdm.write(f"  Saved best EMA weights (meanF1={best_score:.4f}) -> {best_path}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    tqdm.write(f"  Early stopping on fold {fold} at epoch {epoch}")
                    break

        # produce OOF from best model
        model.load_state_dict(torch.load(best_path, map_location=DEVICE))
        model.eval()
        va_loader_eval = DataLoader(va_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)
        all_preds, all_probs = [], []
        with torch.no_grad():
            for imu, thm, tof, y in va_loader_eval:
                imu = imu.to(DEVICE)
                thm = thm.to(DEVICE) if thm.numel() else torch.zeros((imu.size(0), imu.size(1), 0), device=DEVICE)
                tof = tof.to(DEVICE) if tof.numel() else torch.zeros((imu.size(0), imu.size(1), 0), device=DEVICE)
                logits = model(imu, thm, tof)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = probs.argmax(axis=1)
                all_probs.append(probs)
                all_preds.extend(preds.tolist())
        all_probs = np.concatenate(all_probs, axis=0)
        oof_pred[va_idx] = np.array(all_preds)
        oof_prob[va_idx] = all_probs
        fold_ckpts.append(best_path)

    # CV metrics
    metrics = competition_metric(train_seq["gesture_idx"].values, oof_pred, idx_to_class, gesture_to_type)
    print(f"\nCV macroF1={metrics['macro_f1']:.4f} | binF1={metrics['binary_f1']:.4f} | meanF1={metrics['mean_f1']:.4f}")

    # Save meta for inference
    cfg_dict = asdict(CFG)
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

# ========== Inference ensemble ==========
class InferenceEnsemble:
    def __init__(self, ckpt_paths: List[str], imu_dim: int, thm_dim: int, tof_dim: int, n_classes: int, device=DEVICE):
        self.device = device
        self.models = []
        for p in ckpt_paths:
            m = CMIModel(imu_dim, thm_dim, tof_dim, n_classes,
                         feat_dim=500,
                         imu1_channels=219, imu1_layers=0, imu1_dropout=0.2946731587132302, imu2_layers=0, imu2_dropout=0.2697745571929592,
                         thm1_channels=82, thm1_dropout=0.2641274454844602, thm2_dropout=0.302896343020985,
                         tof1_channels=82, tof1_dropout=0.2641274454844602, tof2_dropout=0.3028963430209852,
                         bert_layers=8, bert_heads=10,
                         cls1_channels=937, cls2_channels=303, cls1_dropout=0.2281834512100508, cls2_dropout=0.22502521933558461
                         ).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            m.eval()
            self.models.append(m)

    @torch.no_grad()
    def predict_proba_batch(self, imu, thm, tof) -> np.ndarray:
        logits_list = []
        for m in self.models:
            logits = m(imu.to(self.device), thm.to(self.device), tof.to(self.device))
            logits_list.append(logits)
        avg_logits = torch.stack(logits_list).mean(0)
        probs = torch.softmax(avg_logits, dim=1).cpu().numpy()
        return probs

    def predict_label_for_sequence_df(self, seq_df: pd.DataFrame) -> str:
        # produce (1, L, C) tensors using SequenceDataset preprocessing code
        seq_df = seq_df.sort_values("sequence_counter").reset_index(drop=True)
        # reuse SequenceDataset._prep_one behavior quickly:
        # IMU
        seq_df = interpolate_fill_nan_safe(seq_df, list(CFG.imu_cols))
        x_imu = seq_df[list(CFG.imu_cols)].astype(np.float32).values
        x_imu = resample_to_fixed_length(x_imu, CFG.max_len)
        x_imu = zscore_per_sequence(x_imu).astype(np.float32)
        # THM
        thm_cols = list(CFG.thm_cols)
        if thm_cols:
            has_thm = any(c in seq_df.columns for c in thm_cols) and not seq_df[thm_cols].isna().all().all()
            if has_thm:
                seq_df = interpolate_fill_nan_safe(seq_df, thm_cols)
                x_thm = seq_df[thm_cols].astype(np.float32).values
                x_thm = resample_to_fixed_length(x_thm, CFG.max_len)
                x_thm = zscore_per_sequence(x_thm).astype(np.float32)
            else:
                x_thm = np.zeros((CFG.max_len, len(thm_cols)), dtype=np.float32)
        else:
            x_thm = np.zeros((CFG.max_len, 0), dtype=np.float32)
        # ToF
        tof_feats = []
        for sensor_id in range(1, CFG.tof_sensors + 1):
            tof_cols = [f"tof_{sensor_id}_v{i}" for i in range(CFG.tof_pixels)]
            existing = [c for c in tof_cols if c in seq_df.columns]
            if existing and not seq_df[existing].replace(-1, np.nan).isna().all().all():
                tof_data = seq_df[existing].replace(-1, np.nan).mean(axis=1).fillna(0).values.reshape(-1,1)
                tof_data = resample_to_fixed_length(tof_data, CFG.max_len)
                tof_data = zscore_per_sequence(tof_data).astype(np.float32)
                tof_feats.append(tof_data[:,0])
            else:
                tof_feats.append(np.zeros(CFG.max_len, dtype=np.float32))
        x_tof = np.stack(tof_feats, axis=1) if len(tof_feats)>0 else np.zeros((CFG.max_len,0), dtype=np.float32)

        imu_t = torch.tensor(x_imu[None,...], dtype=torch.float32)
        thm_t = torch.tensor(x_thm[None,...], dtype=torch.float32)
        tof_t = torch.tensor(x_tof[None,...], dtype=torch.float32)
        probs = self.predict_proba_batch(imu_t, thm_t, tof_t)[0]
        pred_idx = int(np.argmax(probs))
        return idx_to_class[pred_idx]

# ========== predict test and save ==========
@torch.no_grad()
def predict_test_and_save_csv():
    meta = json.load(open(os.path.join(OUTPUT_DIR, "meta.json")))
    ckpts = meta["fold_ckpts"]

    # infer dims from SequenceDataset first sample
    probe_rows = train_seq.head(1)
    probe_ds = SequenceDataset(train_data, probe_rows, CFG.imu_cols, CFG.max_len, labels=None, use_all_sensors=CFG.use_all_sensors)
    sample_imu, sample_thm, sample_tof, _ = probe_ds[0]
    imu_dim = sample_imu.shape[1]
    thm_dim = sample_thm.shape[1] if sample_thm.numel() else 0
    tof_dim = sample_tof.shape[1] if sample_tof.numel() else 0

    ens = InferenceEnsemble(ckpts, imu_dim, thm_dim, tof_dim, len(idx_to_class), device=DEVICE)

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

# ========== main ==========
if __name__ == "__main__":
    fold_ckpts = run_cv_and_train_full()
    predict_test_and_save_csv()
