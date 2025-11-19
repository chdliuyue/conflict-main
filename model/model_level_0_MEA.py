import os
import argparse
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


# Because evaluate_multitask_predictions and format_results_table are unused,
# we can keep this import commented out (unless you intend to re-enable it).
# from module.metrics import evaluate_multitask_predictions, format_results_table


# Read and prepare data
def load_data(train_path, test_path):
    # Load the training and testing data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    feature_names = ['UF', 'UAS', 'UD', 'UAL', 'DF', 'DAS', 'DD', 'DAL', 'rq_rel', 'rk_rel', 'CV_v', 'E_BRK']

    # Extract the selected features and labels
    X_train = train_data[feature_names].values
    X_test = test_data[feature_names].values

    y_train = train_data.iloc[:, -3:].values  # Labels (last three columns)
    y_test = test_data.iloc[:, -3:].values  # Labels (last three columns)

    return X_train, y_train, X_test, y_test, feature_names


# Build and train PPO ordered-logit models
def train_ppo_model(X, y, feature_names, output_dir, significance_threshold=0.10):
    """
    Train Partial Proportional Odds (PPO) ordered-logit models and store the coefficient tables.
    """

    models = []
    coef_matrix = []
    task_names = ['TTC', 'DRAC', 'PSD']  # Label names per task

    for task in range(y.shape[1]):  # Train a separate model for each task (TTC, DRAC, PSD)
        print(f"Training model for task {task_names[task]}...")

        # Prepare labels for the PPO formulation
        y_task = y[:, task]

        # Inspect the class cardinality
        unique_classes = np.unique(y_task)
        print(f"Task {task_names[task]} has {len(unique_classes)} classes: {unique_classes}")

        # Train the OrderedModel
        # Fix: removed exog_names=feature_names because it raised AttributeError
        model = OrderedModel(y_task, X, distr='probit')
        result = model.fit(method='bfgs')
        models.append(result)

        # Fix: retrieve every parameter name (features plus thresholds/intercepts)
        # all_param_names = [row[0] for row in result.summary().tables[0].data[1:]] # <-- offending line

        # --- Revised approach ---
        # result.params is an ndarray containing feature coefficients followed by thresholds
        # 1. Count the features
        num_features = len(feature_names)  # Provided by load_data

        # 2. Total number of parameters
        num_params = len(result.params)

        # 3. Number of thresholds
        num_thresholds = num_params - num_features
        if num_thresholds < 0:
            # Should never occur, but keep a safety check
            raise ValueError(
                f"Fewer parameters ({num_params}) than feature names ({num_features})? This shouldn't happen.")

        # 4. Dynamically build names for the thresholds using simple placeholders
        threshold_names = [f'threshold_{i + 1}' for i in range(num_thresholds)]

        # 5. Combine them into a complete list (feature coefficients first, thresholds second)
        all_param_names = feature_names + threshold_names

        # 6. Sanity-check lengths
        if len(all_param_names) != num_params:
            # Again, this should never happen
            raise ValueError(
                f"Fatal error: Parameter name list length ({len(all_param_names)}) "
                f"does not match result.params length ({num_params})."
            )

        # all_param_names now matches result.params in length
        params_series = pd.Series(result.params, index=all_param_names)
        pvalues_series = pd.Series(result.pvalues, index=all_param_names)

        # Slice out only the feature coefficients (.loc now works reliably)
        feature_coef_df = pd.DataFrame({
            'coef': params_series.loc[feature_names],
            'p_value': pvalues_series.loc[feature_names]
        })

        # Null coefficients with insignificant p-values
        feature_coef_df['coef'] = np.where(feature_coef_df['p_value'] > significance_threshold, np.nan,
                                           feature_coef_df['coef'])
        coef_matrix.append(feature_coef_df['coef'].values)

        print(result.summary())

    # Persist the coefficient tables for every task into a CSV file
    coef_matrix = np.array(coef_matrix).T
    coef_matrix_df = pd.DataFrame(coef_matrix, columns=task_names, index=feature_names)

    coef_matrix_df.to_csv(
        os.path.join(output_dir, 'level_0_coefficients.csv'), index_label='feature'
    )
    print(f"Level 0 coefficients saved to {output_dir}/level_0_coefficients.csv")

    return models


# --- Evaluation helpers were removed as requested ---

# --- Added capability: marginal-effects analysis ---
def analyze_marginal_effects(models, X_train, feature_names, task_names, feature_to_analyze, output_dir):
    """
    Analyze and plot how each class probability changes as one feature varies
    while all other features stay fixed at their means.
    """
    print(f"\nAnalyzing marginal effects for feature: {feature_to_analyze}...")

    try:
        # Look up the index of the target feature
        feature_index = feature_names.index(feature_to_analyze)
    except ValueError:
        print(f"Error: Feature '{feature_to_analyze}' not found in feature list.")
        print(f"Available features: {feature_names}")
        return

    # 1. Compute the mean across every feature
    X_means = np.mean(X_train, axis=0)

    # 2. Determine the observed range for the selected feature
    f_min = np.min(X_train[:, feature_index])
    print(f_min)
    f_max = np.max(X_train[:, feature_index])
    print(f_max)
    # Create a smooth range with 100 evenly spaced points
    feature_range = np.linspace(f_min, f_max, 100)

    for i, (model, task_name) in enumerate(zip(models, task_names)):
        print(f"Generating plot for task: {task_name}")

        # 3. Build a "synthetic" dataset
        #    - shape = (100, num_features)
        #    - every row equals X_means
        X_synthetic = np.tile(X_means, (100, 1))

        # 4. Replace the analyzed feature column with feature_range
        X_synthetic[:, feature_index] = feature_range

        # 5. Predict probabilities; result shape is (100, num_classes)
        probas = model.predict(X_synthetic)

        if np.isnan(probas).any():
            print(f"Warning: NaN values encountered in predictions for task {task_name}. Skipping plot.")
            continue

        num_classes = probas.shape[1]
        print(f"Task {task_name} has {num_classes} outcome classes for plotting.")

        # 6. Plot the marginal effect
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams.update({
            'axes.linewidth': 2,
            'image.cmap': 'vlag'  # Set the default colormap for all plots  # coolwarm
        })

        # --- Added: human-readable class labels ---
        class_labels = {
            0: "Safe (0)",
            1: "Low (1)",
            2: "Medium (2)",
            3: "High (3)"
        }
        # --- End addition ---

        fig, ax = plt.subplots(figsize=(8, 6))
        for c in range(num_classes):
            # Plot the probability curve for each class with the updated labels
            label = class_labels.get(c, f"Class {c}")  # Fall back to "Class c" for >3
            plt.plot(feature_range, probas[:, c], label=label, lw=6)

        # --- Styling tweak: increase font sizes ---
        # plt.xlabel(f"Value of {feature_to_analyze}", fontsize=14)
        # plt.ylabel("Predicted Probability", fontsize=14)
        # plt.title(f"Predicted Probabilities for {task_name} as {feature_to_analyze} Varies", fontsize=16)

        # --- Styling tweak: borderless legend with larger font ---
        plt.legend(frameon=False, fontsize=22)
        ax.tick_params(axis='x', labelsize=24)
        ax.tick_params(axis='y', labelsize=24)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim(0, 1)  # Probabilities stay within [0, 1]

        # 7. Persist the figure
        plot_filename = os.path.join(output_dir, f"marginal_effects_{task_name}_{feature_to_analyze}.png")
        plt.savefig(plot_filename)
        plt.close()  # Release the figure from memory

        print(f"Saved plot: {plot_filename}")


# Main entry point
def main():
    # Configure input/output paths and execution parameters
    ratio_name = "highD_ratio_20"
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=f"../data/{ratio_name}/train.csv")
    ap.add_argument("--test", default=f"../data/{ratio_name}/test.csv")
    ap.add_argument("--train_old", default=f"../data/{ratio_name}/train_old.csv")
    ap.add_argument("--test_old", default=f"../data/{ratio_name}/test_old.csv")
    ap.add_argument("--out_dir", default=f"../output/{ratio_name}/results_level_0_MEA")
    # Added argument: select which feature to analyze
    ap.add_argument("--feature", default="CV_v", help="Feature to analyze for marginal effects (e.g., UF, CV_v)")
    args = ap.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Load the datasets and capture the feature_names
    X_train, y_train, X_test, y_test, feature_names = load_data(args.train_old, args.test_old)

    # Train the PPO models and write the coefficient tables
    models = train_ppo_model(X_train, y_train, feature_names, args.out_dir)

    # --- Added: run marginal-effects analysis ---
    task_names = ['TTC', 'DRAC', 'PSD']  # Keep consistent with train_ppo_model
    analyze_marginal_effects(models, X_train, feature_names, task_names, args.feature, args.out_dir)

    print("\nAnalysis complete. Coefficient CSV and marginal effects plots saved to:")
    print(args.out_dir)


if __name__ == "__main__":
    main()

