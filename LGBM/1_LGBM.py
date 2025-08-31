# %% imports
import os
import json
import random
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
import polars as pl

from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold

import lightgbm as lgb

# %% config
INPUT_DIR = "../kaggle_cmi_2025/data"
OUTPUT_DIR = "./outputs_lgbm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED = 42
def set_seed(seed=RANDOM_SEED):
    random.seed(seed); np.random.seed(seed)
set_seed()

@dataclass
class Config:
    max_len: int = 400
    n_folds: int = 5
    imu_cols: Tuple[str, ...] = ("acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z")

CFG = Config()

# %% load data
train_data = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test_data  = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))

present_cols = [c for c in CFG.imu_cols if c in train_data.columns]
CFG.imu_cols = tuple(present_cols)

# %% preprocessing utils
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
    T, C = arr.shape
    if T == out_len: return arr
    if T <= 1:
        base = np.zeros((out_len, C), dtype=np.float32)
        if T == 1: base[:] = arr[0]
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

def competition_metric(y_true_idx, y_pred_idx, idx_to_class, gesture_to_type):
    macro_f1 = f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
    y_true_type = np.array([1 if gesture_to_type[idx_to_class[t]] == "Target" else 0 for t in y_true_idx])
    y_pred_type = np.array([1 if gesture_to_type[idx_to_class[p]] == "Target" else 0 for p in y_pred_idx])
    bin_f1 = f1_score(y_true_type, y_pred_type, average="binary", zero_division=0)
    return {"macro_f1": macro_f1, "binary_f1": bin_f1, "mean_f1": (macro_f1+bin_f1)/2.0}

# %% sequence info
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

# %% feature extraction
def extract_features(x: np.ndarray) -> np.ndarray:
    feats = []
    for c in range(x.shape[1]):
        sig = x[:, c]
        feats.extend([
            np.mean(sig), np.std(sig), np.min(sig), np.max(sig),
            np.median(sig), np.percentile(sig, 25), np.percentile(sig, 75),
            np.sum(np.abs(np.diff(sig))),
        ])
        fft_vals = np.abs(np.fft.rfft(sig))
        feats.append(np.mean(fft_vals))
        feats.append(np.max(fft_vals))
    return np.array(feats, dtype=np.float32)

def build_feature_table(seq_df: pd.DataFrame, seq_info: pd.DataFrame, imu_cols: Tuple[str, ...], max_len: int):
    X_feats, y = [], []
    for _, row in tqdm(seq_info.iterrows(), total=len(seq_info)):
        sid = row["sequence_id"]
        df = seq_df[seq_df["sequence_id"] == sid].sort_values("sequence_counter")
        df = interpolate_fill_nan_safe(df, imu_cols)
        x = df[list(imu_cols)].astype(np.float32).values
        x = resample_to_fixed_length(x, max_len)
        x = zscore_per_sequence(x)
        feats = extract_features(x)
        X_feats.append(feats)
        if "gesture_idx" in row:
            y.append(row["gesture_idx"])
    X_feats = np.stack(X_feats)
    y = np.array(y) if len(y) else None
    return X_feats, y

# %% CV training
def run_cv_lightgbm():
    X, y = build_feature_table(train_data, train_seq, CFG.imu_cols, CFG.max_len)
    subjects = train_seq["subject"].astype(str).values
    splitter = StratifiedGroupKFold(n_splits=CFG.n_folds, shuffle=True, random_state=RANDOM_SEED)

    oof_pred = np.full(len(y), -1)
    oof_prob = np.zeros((len(y), len(le.classes_)))
    fold_models = []

    for fold, (tr_idx, va_idx) in enumerate(splitter.split(X, y, groups=subjects), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        train_set = lgb.Dataset(X_tr, label=y_tr)
        val_set   = lgb.Dataset(X_va, label=y_va)

        params = {
            "objective": "multiclass",
            "num_class": len(le.classes_),
            "metric": "multi_logloss",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "seed": RANDOM_SEED,
            "class_weight": "balanced",   # ⚡ NEW
        }

        model = lgb.train(
            params,
            train_set,
            valid_sets=[val_set],
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(30),
                lgb.log_evaluation(50)
            ]
        )
        fold_models.append(model)

        prob = model.predict(X_va)
        pred = np.argmax(prob, axis=1)
        oof_pred[va_idx] = pred
        oof_prob[va_idx] = prob

        model_path = os.path.join(OUTPUT_DIR, f"fold{fold}.txt")
        model.save_model(model_path)

        metrics = competition_metric(y_va, pred, idx_to_class, gesture_to_type)
        print(f"Fold {fold} | macroF1 {metrics['macro_f1']:.4f} | binF1 {metrics['binary_f1']:.4f} | meanF1 {metrics['mean_f1']:.4f}")

    metrics = competition_metric(y, oof_pred, idx_to_class, gesture_to_type)
    print(f"\nCV macroF1={metrics['macro_f1']:.4f} | binF1={metrics['binary_f1']:.4f} | meanF1={metrics['mean_f1']:.4f}")

    meta = {
        "cfg": asdict(CFG),
        "classes": idx_to_class,
        "gesture_to_type": gesture_to_type,
    }
    with open(os.path.join(OUTPUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return fold_models

fold_models = run_cv_lightgbm()

# %% inference on test
def predict_test_and_save_csv(fold_models):
    X_test, _ = build_feature_table(test_data, test_seq, CFG.imu_cols, CFG.max_len)
    probs_list = [m.predict(X_test) for m in fold_models]
    probs = np.mean(probs_list, axis=0)
    preds = np.argmax(probs, axis=1)
    labels = [idx_to_class[i] for i in preds]

    sub = pd.DataFrame({"sequence_id": test_seq["sequence_id"], "gesture": labels})
    sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
    sub.to_csv(sub_path, index=False)
    print(f"Saved submission to {sub_path}")

predict_test_and_save_csv(fold_models)
