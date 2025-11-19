# -*- coding: utf-8 -*-
"""
1) Load the CSV (first 12 columns = features, last 3 columns = ordinal labels 0/1/2/3)
2) Perform stratified splits on the joint 3-label distribution (fallback to single-label split)
3) Fit a StandardScaler on the training set only and transform both train/test
4) Save train.csv / test.csv (standardized features + original labels)
5) Print per-label class counts and ratios for all/train/test splits
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def make_joint_code(y_mat: np.ndarray) -> np.ndarray:
    a = y_mat[:, 0].astype(int)
    b = y_mat[:, 1].astype(int)
    c = y_mat[:, 2].astype(int)
    return a * 16 + b * 4 + c  # 0..63

def show_count_ratio(tag: str, y: np.ndarray):
    vals, cnt = np.unique(y, return_counts=True)
    ratio = cnt / len(y) if len(y) > 0 else np.zeros_like(cnt, dtype=float)
    cnt_dict   = {int(v): int(c)   for v, c in zip(vals, cnt)}
    ratio_dict = {int(v): float(r) for v, r in zip(vals, ratio)}
    print(f"[{tag}] class counts: {cnt_dict}")
    print(f"[{tag}] class ratios: {ratio_dict}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="../data/highD/all_windows_10s_clean.csv",
                    help="Input CSV path (first 12 columns features, last 3 labels)")
    ap.add_argument("--train_out", default="../data/highD_ratio_20/train.csv", help="Train-set output CSV")
    ap.add_argument("--test_out", default="../data/highD_ratio_20/test.csv", help="Test-set output CSV")
    ap.add_argument("--train_out_old", default="../data/highD_ratio_20/train_old.csv", help="Train-set (unscaled) CSV")
    ap.add_argument("--test_out_old", default="../data/highD_ratio_20/test_old.csv", help="Test-set (unscaled) CSV")
    ap.add_argument("--test_size", type=float, default=0.20, help="Test split ratio")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    args = ap.parse_args()

    # 1) Load and identify columns
    df = pd.read_csv(args.input)
    if df.shape[1] < 15:
        raise ValueError(f"Insufficient columns: detected {df.shape[1]} but need at least 15 (12 features + 3 labels)")

    feat_cols  = list(df.columns[:12])
    label_cols = list(df.columns[-3:])
    print(f"[INFO] Feature columns (12): {feat_cols}")
    print(f"[INFO] Label columns (3): {label_cols}")

    # 2) Clean: keep required columns, drop NaN/Inf, clamp labels to {0,1,2,3}
    df = df[feat_cols + label_cols].copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feat_cols + label_cols).reset_index(drop=True)
    for lc in label_cols:
        df[lc] = df[lc].astype(int)
        df = df[df[lc].isin([0, 1, 2, 3])]
    df = df.reset_index(drop=True)
    print(f"[INFO] Samples after cleaning: {len(df)}")

    # —— Label counts/ratios for the full dataset —— #
    for lc in label_cols:
        show_count_ratio(f"All-{lc}", df[lc].to_numpy())

    X = df[feat_cols].to_numpy(dtype=float)
    Y = df[label_cols].to_numpy(dtype=int)

    # 3) Stratified split (prefer the joint 3-label distribution)
    y_joint = make_joint_code(Y)
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_idx, te_idx = next(sss.split(X, y_joint))
        how = "joint (3-label) stratified split"
    except ValueError as e:
        print(f"[WARN] Joint stratification failed: {e} → fall back to single-label stratification ({label_cols[0]})")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
        tr_idx, te_idx = next(sss.split(X, Y[:, 0]))
        how = f"fallback: single-label stratified ({label_cols[0]})"
    print(f"[INFO] Stratification: {how} | test_size={args.test_size}")

    X_train, X_test = X[tr_idx], X[te_idx]
    Y_train, Y_test = Y[tr_idx], Y[te_idx]

    # —— Label counts/ratios for train/test —— #
    for j, lc in enumerate(label_cols):
        show_count_ratio(f"Train-{lc}", Y_train[:, j])
        show_count_ratio(f" Test-{lc}", Y_test[:, j])

    # 4) Standardize: fit on training set only, transform both splits
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std  = scaler.transform(X_test)

    # 5) Assemble and write the CSVs
    train_df = pd.DataFrame(X_train_std, columns=feat_cols)
    test_df  = pd.DataFrame(X_test_std,  columns=feat_cols)
    for j, lc in enumerate(label_cols):
        train_df[lc] = Y_train[:, j]
        test_df[lc]  = Y_test[:, j]

    train_df.to_csv(args.train_out, index=False)
    test_df.to_csv(args.test_out, index=False)
    print(f"[DONE] Train CSV: {args.train_out} | Test CSV: {args.test_out}")

    # 6) Save unstandardized versions
    train_df_old = pd.DataFrame(X_train, columns=feat_cols)
    test_df_old = pd.DataFrame(X_test, columns=feat_cols)
    for j, lc in enumerate(label_cols):
        train_df_old[lc] = Y_train[:, j]
        test_df_old[lc]  = Y_test[:, j]
    train_df_old.to_csv(args.train_out_old, index=False)
    test_df_old.to_csv(args.test_out_old, index=False)


if __name__ == "__main__":
    main()
