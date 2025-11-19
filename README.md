# Conflict Risk Modeling Pipeline

This repository processes raw highD trajectory recordings into conflict-focused
features and trains classification models for three surrogate safety metrics
(TTC, DRAC, PSD). The workflow spans data extraction, cleaning, exploratory
summaries, train/test generation, and multiple modeling options (tree-based and
ordered-logit with SHAP/Marginal Effects diagnostics).

## Repository layout

```
data/                     # Intermediate CSVs (clean windows, train/test splits)
datasets/highD/data/      # <-- place the original highD recordings here
model/                    # Modeling scripts (XGBoost, ordered logit, SHAP, MEA)
module/metrics.py         # Shared evaluation helpers (accuracy/F1/QWK/etc.)
processData/              # End-to-end pipeline and utility scripts
```

## Preparing the data

1. **Place the raw highD dataset** (all `XX_tracks.csv`, `XX_tracksMeta.csv`,
   and `XX_recordingMeta.csv` files) under `datasets/highD/data/`. The default
   arguments in `processData/process_highD.py` expect this layout.

2. **Generate lane-level windows and clean features**

   ```bash
   python processData/process_highD.py \
       --root ../datasets/highD/data/ \
       --out ../data/highD/
   ```

   `process_highD.py` wraps `highd_pipeline.py` to:

   * read every recording, infer lane directions, smooth kinematics, and compute
     surrogate-safety metrics per 10-second window (configurable window/stride).
   * aggregate features such as UF/DF flows, density consistency checks,
     rq/rk relative gradients, speed variation, braking exposure, jerk ratios,
     and node-level TTC/DRAC/PSD scores.
   * optionally clean the 12 core features + 3 ordinal labels, producing
     `data/highD/all_windows_10s_clean.csv`.
   * (optional) print/save summaries via `summarize_windows.py`.

   Key toggles include reaction-time compensation (`--ssm_tau_*`), geometric
   shrink parameters, jerk thresholds, worker count, and whether to skip the
   cleaning/summary steps.

3. **(Optional) Clean an existing CSV**

   Use `clean_conflict_dataset.py` when you already have a concatenated windows
   CSV and want to enforce the 12-feature/3-label schema without rerunning the
   pipeline:

   ```bash
   python processData/clean_conflict_dataset.py \
       --input ../data/highD/all_windows_10s.csv \
       --output ../data/highD/all_windows_10s_clean.csv
   ```

   The script validates feature/label columns, drops invalid rows, reports any
   missing columns/out-of-range labels, and (optionally) prints summary tables.

4. **Split into train/test sets**

   ```bash
   python processData/split_data.py \
       --input ../data/highD/all_windows_10s_clean.csv \
       --train_out ../data/highD_ratio_20/train.csv \
       --test_out ../data/highD_ratio_20/test.csv
   ```

   This script:

   * keeps the 12 features + 3 labels, dropping rows with NaN/Inf or labels
     outside {0,1,2,3}.
   * prints per-label class counts/ratios for the full dataset and each split.
   * performs a stratified split on the joint (TTC, DRAC, PSD) label
     combination, falling back to single-label stratification if necessary.
   * fits a `StandardScaler` on the training features only, applies it to both
     splits, and saves both standardized (`train.csv`, `test.csv`) and raw
     (`train_old.csv`, `test_old.csv`) versions under `data/highD_ratio_20/`.

5. **Summaries and diagnostics**

   * `summarize_windows.py` prints/saves aggregate statistics for features,
     masks, and label distributions (useful before modeling).
   * `proc_utils.py` hosts common helpers (column resolution, smoothing,
     crossing detectors, etc.) used by the pipeline.

## Modeling scripts

All modeling scripts assume the standardized train/test CSVs produced in the
previous step (paths default to `../data/highD_ratio_20/`).

* `model/model_XGBoost.py`
  * Loads the data, selects 12 engineered features, and trains one multi-class
    XGBoost classifier per task (TTC, DRAC, PSD) with grid search.
  * Saves feature-importance tables (`xgb_importances.csv`) and evaluates the
    models using `module.metrics` (accuracy, F1, QWK, OrdMAE, NLL, Brier, AUROC,
    BrdECE). Results are logged to `evaluation_results.txt`.

* `model/model_XGBoost_shap.py`
  * Repeats the XGBoost training (optionally with a broader hyper-parameter
    grid) and then runs SHAP analysis on the test set.
  * Outputs dependence plots for every feature/class and (commented) examples of
    beeswarm/donut charts for feature importances per class.

* `model/model_level_0.py`
  * Fits Partial Proportional Odds (ordered logit) models per task using
    `statsmodels.OrderedModel`.
  * Exports coefficient tables (`level_0_coefficients.csv`) and evaluates the
    models with the same metrics as the XGBoost variant.

* `model/model_level_0_MEA.py`
  * Extends the PPO ordered-logit modeling with a curated feature list and
    enhanced coefficient extraction.
  * Adds marginal effects analysis: specify `--feature CV_v` (or any feature) to
    visualize how class probabilities evolve as that feature changes while all
    others remain at their means. Plots are saved as
    `marginal_effects_<task>_<feature>.png`.

## Typical end-to-end flow

1. Place the original highD dataset under `datasets/highD/data/`.
2. Run `processData/process_highD.py` to build per-window features and obtain
   `data/highD/all_windows_10s_clean.csv`.
3. Split into train/test via `processData/split_data.py`.
4. Train/evaluate models with the scripts under `model/` (e.g.
   `python model/model_XGBoost.py --train ../data/highD_ratio_20/train.csv \
   --test ../data/highD_ratio_20/test.csv`).
5. Use `model/model_XGBoost_shap.py` for interpretability or
   `model/model_level_0_MEA.py` for marginal-effect plots.

Adjust any paths/thresholds via the provided CLI arguments to match your local
layout or desired experimental setup.
