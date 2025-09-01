# %% imports
import os
import json
from typing import Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import lightgbm as lgb

# %% config
INPUT_DIR = "../kaggle_cmi_2025/data"
OUTPUT_DIR = "./outputs_lgbm_thm"
N_FOLDS = 5
MAX_LEN = 1000

# %% preprocessing utils
def interpolate_fill_nan_safe(df: pd.DataFrame, cols: Tuple[str, ...]) -> pd.DataFrame:
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
    X_feats = []
    for _, row in tqdm(seq_info.iterrows(), total=len(seq_info)):
        sid = row["sequence_id"]
        df = seq_df[seq_df["sequence_id"] == sid].sort_values("sequence_counter")
        df = interpolate_fill_nan_safe(df, imu_cols)
        x = df[list(imu_cols)].astype(np.float32).values
        x = resample_to_fixed_length(x, max_len)
        x = zscore_per_sequence(x)
        feats = extract_features(x)
        X_feats.append(feats)
    X_feats = np.stack(X_feats)
    return X_feats

# %% load test data
test_data = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
test_seq = test_data.groupby("sequence_id", sort=False).agg(n_rows=("row_id", "count")).reset_index()

# %% load meta
with open(os.path.join(OUTPUT_DIR, "meta.json"), "r") as f:
    meta = json.load(f)
imu_cols = tuple(meta["cfg"]["imu_cols"])
idx_to_class = meta["classes"]

# %% load models
fold_models = []
for fold in range(1, N_FOLDS+1):
    model_file = os.path.join(OUTPUT_DIR, f"fold{fold}.txt")
    model = lgb.Booster(model_file=model_file)
    fold_models.append(model)

# %% predict
X_test = build_feature_table(test_data, test_seq, imu_cols, MAX_LEN)
probs_list = [m.predict(X_test) for m in fold_models]
probs = np.mean(probs_list, axis=0)
preds = np.argmax(probs, axis=1)
labels = [idx_to_class[i] for i in preds]

# %% save submission
submission = pd.DataFrame({"sequence_id": test_seq["sequence_id"], "gesture": labels})
submission_path = os.path.join(OUTPUT_DIR, "submission_inference.csv")
submission.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path}")
