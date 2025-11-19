import os
import argparse
import numpy as np
import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from module.metrics import evaluate_multitask_predictions, format_results_table


# Read and prepare data
def load_data(train_path, test_path):
    # Load the training and testing data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Extract features (first 12 columns) and labels (last three columns)
    X_train = train_data.iloc[:, :-3].values
    y_train = train_data.iloc[:, -3:].values
    X_test = test_data.iloc[:, :-3].values
    y_test = test_data.iloc[:, -3:].values

    return X_train, y_train, X_test, y_test

# Build and train the ordered logit models
def train_ppo_model(X, y, output_dir, significance_threshold=0.10):
    """
    Train partial proportional-odds ordered logit models and persist the coefficient tables.
    """

    models = []
    coef_matrix = []
    task_names = ['TTC', 'DRAC', 'PSD']  # Label names for each task
    for task in range(y.shape[1]):  # Train a model for every task (TTC, DRAC, PSD)
        print(f"Training model for task {task_names[task]}...")

        # Prepare task-specific labels
        y_task = y[:, task]

        # Train the OrderedModel instance
        model = OrderedModel(y_task, X, distr='probit')
        result = model.fit(method='bfgs')
        models.append(result)

        # Capture the coefficient table and collect it
        coef_df = pd.DataFrame({
            'feature': model.exog_names,  # Feature names
            'coef': result.params,        # Coefficients
            'p_value': result.pvalues     # p-values
        })

        # Null coefficients whose p-values do not pass the threshold
        coef_df['coef'] = np.where(coef_df['p_value'] > significance_threshold, np.nan, coef_df['coef'])
        coef_matrix.append(coef_df['coef'].values)

        print(result.summary())

    # Persist the coefficients for all tasks in a CSV file
    coef_matrix = np.array(coef_matrix).T
    feature_names = coef_df['feature'].values
    coef_matrix_df = pd.DataFrame(coef_matrix, columns=task_names, index=feature_names)

    coef_matrix_df.to_csv(
        os.path.join(output_dir, 'level_0_coefficients.csv'), index_label='feature'
    )
    print(f"Level 0 coefficients saved to {output_dir}/level_0_coefficients.csv")

    return models

# Evaluate the models
def evaluate_model(models, X_test, y_test):
    """
    Evaluate the PPO model performance for each task.
    """
    task_names = ['TTC', 'DRAC', 'PSD']
    all_probas = []
    for task, model in enumerate(models):
        print(f"Evaluating model for task {task_names[task]}...")
        probas = np.asarray(model.predict(X_test))
        if probas.ndim != 2:
            raise ValueError(f"Expected 2-D probability array, got shape {probas.shape}")
        all_probas.append(probas)

    all_probas = np.stack(all_probas, axis=1)  # [N, M, C]
    metrics = evaluate_multitask_predictions(y_true=y_test, probas=all_probas, task_names=task_names)

    print(format_results_table(metrics))

    return metrics

# Main entry point
def main():
    # Configure input/output paths and other parameters
    ratio_name = "highD_ratio_20"
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../data/" + ratio_name + "/train.csv")
    ap.add_argument("--test", default="../data/" + ratio_name + "/test.csv")
    ap.add_argument("--out_dir", default="../output/" + ratio_name + "/results_level_0")
    args = ap.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Load the data
    X_train, y_train, X_test, y_test = load_data(args.train, args.test)

    # Train the PPO models and save the coefficient tables
    models = train_ppo_model(X_train, y_train, args.out_dir)

    # Evaluate the models
    metrics = evaluate_model(models, X_test, y_test)

    # Save the evaluation results
    results_file = os.path.join(args.out_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        # Write the table header
        f.write("Task | Accuracy | F1 | QWK | OrdMAE | NLL | Brier | AUROC | BrdECE\n")
        # Output the metrics for each task
        for metric in metrics:
            f.write(
                f"{metric.task} | {metric.accuracy:.4f} | {metric.f1_score:.4f} | "
                f"{metric.qwk:.4f} | {metric.ordmae:.4f} | {metric.nll:.4f} | "
                f"{metric.brier:.4f} | {metric.auroc:.4f} | {metric.brdece:.4f}\n"
            )

    print(f"Evaluation results saved to {results_file}")

if __name__ == "__main__":
    main()
