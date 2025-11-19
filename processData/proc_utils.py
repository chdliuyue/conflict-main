# proc_utils.py
# -*- coding: utf-8 -*-
"""
Auxiliary data-processing helpers used alongside the TTC / DRAC / PSD routines.
These utilities were factored out of process_highD and keep the same names/signatures
to minimize the required changes.

Units: distance in metres, speed in m/s, acceleration in m/s², time in seconds.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

__all__ = [
    "percentile",
    "resolve_col",
    "normalize_vehicle_dims",
    "add_time_and_dt",
    "smooth_series",
    "lane_direction_map",
    "infer_lane_inc",
    "crossing_events_full",
    "precompute_location_anchors",
    "parse_start_time_seconds",
    "natural_sorted_dirs",
    "concat_outputs",
]

# ------------------------- Constants ------------------------- #
CROSS_HYSTERESIS_M: float = 0.2   # Hysteresis distance (m) when checking section crossings
EPS: float = 1e-12


# ------------------------- Core / generic helpers ------------------------- #
def percentile(s: pd.Series, q_percent: float) -> float:
    """
    Compute the percentile using only finite values (q is the 0–100 percentile).
    """
    s = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(np.percentile(s, q_percent)) if len(s) else np.nan


def resolve_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return the first matching column from df.columns using case-insensitive lookup,
    preserving the original casing.
    """
    m = {c.lower(): c for c in df.columns}
    for name in candidates:
        c = m.get(name.lower())
        if c is not None:
            return c
    return None


def normalize_vehicle_dims(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize length/width fields from heterogeneous sources to:
      - veh_len: vehicle length (m)
      - veh_wid: vehicle width (m)
    Prefer length/height and fall back to width when missing.
    """
    out = df
    if "length" in out.columns:
        out["veh_len"] = pd.to_numeric(out["length"], errors="coerce")
    elif "width" in out.columns:
        out["veh_len"] = pd.to_numeric(out["width"], errors="coerce")
    else:
        out["veh_len"] = np.nan

    if "height" in out.columns:
        out["veh_wid"] = pd.to_numeric(out["height"], errors="coerce")
    elif "width" in out.columns:
        out["veh_wid"] = pd.to_numeric(out["width"], errors="coerce")
    else:
        out["veh_wid"] = np.nan

    return out


def add_time_and_dt(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    """
    Derive absolute time and per-frame dt (seconds) from frame and fps.
    """
    df["time"] = (df["frame"] - 1) / float(fps)
    df["dt"] = 1.0 / float(fps)
    return df


def smooth_series(x: pd.Series, win_frames: int = 10) -> pd.Series:
    """
    Apply a centered moving average with an odd window; return the original series if too short.
    """
    w = max(3, int(win_frames) | 1)
    if len(x) < w:
        return x
    return x.rolling(window=w, center=True, min_periods=max(1, w // 2)).mean()


# ------------------------- Direction / lane utilities ------------------------- #
def lane_direction_map(tracks: pd.DataFrame, tracks_meta: pd.DataFrame) -> Dict[int, int]:
    """
    Estimate the dominant driving direction per laneId (mode of tracksMeta.drivingDirection).
    Returns {laneId -> drivingDirection}.
    """
    id2dir = dict(zip(tracks_meta["id"].values, tracks_meta["drivingDirection"].values))
    tmp = tracks[["laneId", "id"]].copy()
    tmp["dir"] = tmp["id"].map(id2dir)
    mode_dir = tmp.groupby("laneId")["dir"].agg(lambda s: s.value_counts().idxmax() if len(s) else np.nan)
    return {int(k): int(v) for k, v in mode_dir.dropna().items()}


def infer_lane_inc(df_lane_raw: pd.DataFrame) -> bool:
    """
    Infer whether the downstream direction corresponds to increasing x.
    Returns True if x increases downstream (inc), False otherwise.
    """
    df_l_sorted = df_lane_raw.sort_values(["id", "frame"])
    try:
        delta_by_id = df_l_sorted.groupby("id")["x"].apply(lambda s: s.iloc[-1] - s.iloc[0]).dropna()
        return bool(float(np.nanmedian(delta_by_id.values)) > 0)
    except Exception:
        return True  # Fallback: assume x increases downstream


# ------------------------- Crossing events / anchors ------------------------- #
def crossing_events_full(
    df_lane: pd.DataFrame,
    X_line: float,
    inc: bool,
    eps_m: float = CROSS_HYSTERESIS_M,
) -> pd.DataFrame:
    """
    Identify section-crossing events with hysteresis for UF/DF and boundary-speed stats.
    Args:
      - df_lane: single-lane frame-level data with columns ["time","id","x_prev","x","xVelocity","v_abs","veh_len"]
      - X_line: x-position of the section
      - inc: True if downstream is positive x
      - eps_m: hysteresis distance (m) to suppress noisy crossings
    Returns rows containing crossing instants ordered by time.
    """
    prev = df_lane["x_prev"] - X_line
    now = df_lane["x"] - X_line
    if inc:
        crossed = ((prev <= 0) & (now >= 0)) | ((prev < -eps_m) & (now > eps_m))
    else:
        crossed = ((prev >= 0) & (now <= 0)) | ((prev > eps_m) & (now < -eps_m))
    cols = ["time", "id", "xVelocity", "v_abs", "veh_len"]
    cols = [c for c in cols if c in df_lane.columns]
    ev = df_lane.loc[crossed, cols].copy()
    return ev.sort_values("time")


def precompute_location_anchors(rec_dirs: List[Path]) -> Dict[int, Tuple[float, float]]:
    """
    Scan every recording to capture the 1%/99% quantiles of x, and use their medians
    as upstream/downstream anchors per location. Returns {locationId -> (x_lo_loc, x_hi_loc)}.
    """
    anchors_raw: Dict[int, List[Tuple[float, float]]] = {}
    for d in rec_dirs:
        meta_fp = d / f"{d.name}_recordingMeta.csv"
        tracks_fp = d / f"{d.name}_tracks.csv"
        tmeta_fp = d / f"{d.name}_tracksMeta.csv"
        if not (meta_fp.exists() and tracks_fp.exists() and tmeta_fp.exists()):
            root = d.parent
            meta_fp = root / f"{d.name}_recordingMeta.csv"
            tracks_fp = root / f"{d.name}_tracks.csv"
            tmeta_fp = root / f"{d.name}_tracksMeta.csv"
            if not (meta_fp.exists() and tracks_fp.exists() and tmeta_fp.exists()):
                continue

        rec_meta = pd.read_csv(meta_fp)
        locationId = int(rec_meta.loc[0, "locationId"])
        try:
            x = pd.read_csv(tracks_fp, usecols=["x"])["x"].dropna().astype(float)
        except Exception:
            df_t = pd.read_csv(tracks_fp)
            col_x = resolve_col(df_t, ["x", "xCenter"])
            x = pd.to_numeric(df_t[col_x], errors="coerce").dropna().astype(float) if col_x else pd.Series(dtype=float)
        if len(x) == 0:
            continue
        x_lo = float(x.quantile(0.01))
        x_hi = float(x.quantile(0.99))
        anchors_raw.setdefault(locationId, []).append((x_lo, x_hi))

    anchors: Dict[int, Tuple[float, float]] = {}
    for loc, pairs in anchors_raw.items():
        los = np.array([p[0] for p in pairs], float)
        his = np.array([p[1] for p in pairs], float)
        anchors[loc] = (float(np.median(los)), float(np.median(his)))
    return anchors


# ------------------------- Time / file utilities ------------------------- #
def parse_start_time_seconds(rec_meta: pd.DataFrame) -> Optional[float]:
    """
    Parse the absolute start time (seconds) from recordingMeta.
    Supports timestamps in seconds/milliseconds or ISO strings. Returns None on failure.
    """
    cand = [c for c in rec_meta.columns if ("start" in c.lower() and ("time" in c.lower() or "stamp" in c.lower()))]
    for c in cand:
        v = rec_meta.loc[0, c]
        if pd.isna(v):
            continue
        try:
            f = float(v)
            if f > 1e12:  # ms
                return f / 1000.0
            if f > 1e9:   # s
                return f
        except Exception:
            ts = pd.to_datetime(v, utc=True, errors="coerce")
            if ts is not pd.NaT:
                return float(ts.timestamp())
    return None


def natural_sorted_dirs(root: Path) -> List[Path]:
    """
    Return two-digit subdirectories (00..99) if present, otherwise infer from *_recordingMeta.csv files.
    """
    if root.exists():
        subdirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 2]
    else:
        subdirs = []
    if subdirs:
        return sorted(subdirs, key=lambda p: int(p.name))
    metas = sorted(root.glob("[0-9][0-9]_recordingMeta.csv"), key=lambda p: int(p.name[:2]))
    ids = [m.name[:2] for m in metas]
    return [root / i for i in ids]


def concat_outputs(out_dir: Path, window_sec: float) -> Path:
    """
    Merge every per-recording output into a single CSV: all_windows_{window}s.csv
    """
    parts = sorted(out_dir.glob(f"[0-9][0-9]_windows_{int(window_sec)}s.csv"),
                   key=lambda p: int(p.name[:2]))
    out_fp = out_dir / f"all_windows_{int(window_sec)}s.csv"
    if not parts:
        return out_fp
    dfs = [pd.read_csv(p) for p in parts]
    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(out_fp, index=False)
    return out_fp
