import os
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import sys
from sklearn.model_selection import GridSearchCV

# Ensure modules can be imported from the parent directory
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from module.metrics import evaluate_multitask_predictions, format_results_table


# Read and prepare data
def load_data(train_path, test_path):
    """
    Read the training/testing data and return NumPy arrays plus feature names.
    """
    # Load training and testing data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Retrieve feature names before converting to NumPy arrays
    # feature_names = train_data.columns[:-3].tolist() # Select features here
    feature_names = ['UF', 'UAS', 'UD', 'UAL', 'DF', 'DAS', 'DD', 'DAL', 'rq_rel', 'rk_rel', 'CV_v', 'E_BRK']

    # Extract features and labels and convert to NumPy arrays
    # X_train = train_data.iloc[:, :-3].values
    # X_test = test_data.iloc[:, :-3].values
    X_train = train_data[feature_names].values
    X_test = test_data[feature_names].values

    y_train = train_data.iloc[:, -3:].values
    y_test = test_data.iloc[:, -3:].values

    return X_train, y_train, X_test, y_test, feature_names


# Build and train the XGBoost models
def train_xgboost_model(X, y, output_dir, feature_names):
    """
    Train a task-specific XGBoost classifier via grid search and export feature importances.
    """
    models = []
    feature_importances_matrix = []
    task_names = ['TTC', 'DRAC', 'PSD']  # Label names for each task

    # Define the hyper-parameter grid
    # This grid is only an example; adjust the ranges as needed
    # param_grid = {
    #     'max_depth': [3, 4, 5],
    #     'learning_rate': [0.05, 0.075, 0.1],
    #     'n_estimators': [100, 150, 200],
    #     'reg_lambda': [0.5, 1.0, 1.5, 2.0]  # L2 regularization
    # }
    param_grid = {
        'max_depth': [3],
        'learning_rate': [0.1],
        'n_estimators': [100],
        'reg_lambda': [1.0]  # L2 regularization
    }

    for task_idx, task_name in enumerate(task_names):
        print(f"--- Starting GridSearchCV for task: {task_name} ---")

        # Prepare the training labels for the current task
        y_task = y[:, task_idx]

        # Initialize the base model (use_label_encoder removed)
        model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            random_state=42
        )

        # Configure GridSearchCV (cv=3 means 3-fold CV, n_jobs=-1 uses all CPU cores)
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring='accuracy',
            cv=3,
            verbose=2,
            n_jobs=-1
        )

        # Execute the grid search
        grid_search.fit(X, y_task)

        print(f"Best parameters found for task {task_name}: {grid_search.best_params_}")

        # Capture the best-performing model
        best_model = grid_search.best_estimator_
        models.append(best_model)

        # Collect and store the feature importances
        importances = best_model.feature_importances_
        feature_importances_matrix.append(importances)

    # Persist the feature importances for all tasks as a single CSV file
    feature_importances_matrix = np.array(feature_importances_matrix).T

    feature_importances_df = pd.DataFrame(
        feature_importances_matrix,
        columns=task_names,
        index=feature_names
    )

    # Save the feature-importance file
    output_path = os.path.join(output_dir, 'xgb_importances.csv')
    feature_importances_df.to_csv(output_path, index_label='feature')
    print(f"Level 0 feature importances saved to {output_path}")

    return models


# Evaluate the models
def evaluate_model(models, X_test, y_test):
    """
    Evaluate the XGBoost performance for each task.
    """
    task_names = ['TTC', 'DRAC', 'PSD']
    all_probas = []

    for i, model in enumerate(models):
        task_name = task_names[i]
        print(f"Evaluating model for task {task_name}...")

        # Use predict_proba to obtain the class probabilities
        probas = model.predict_proba(X_test)

        if probas.ndim != 2:
            raise ValueError(f"Expected 2-D probability array, got shape {probas.shape}")

        all_probas.append(probas)

    # Stack every task's probabilities into an [N, M, C] tensor
    all_probas = np.stack(all_probas, axis=1)

    # Compute metrics with the shared evaluation helper
    metrics = evaluate_multitask_predictions(y_true=y_test, probas=all_probas, task_names=task_names)

    print(format_results_table(metrics))

    return metrics


# Main entry point
def main():
    # Configure paths and runtime parameters
    ratio_name = "highD_ratio_20"
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=f"../data/{ratio_name}/train_old.csv")
    ap.add_argument("--test", default=f"../data/{ratio_name}/test_old.csv")
    ap.add_argument("--out_dir", default=f"../output/{ratio_name}/results_xgb_gs")
    args = ap.parse_args()

    # Ensure the output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    # Load the data and recover the feature names
    X_train, y_train, X_test, y_test, feature_names = load_data(args.train, args.test)

    # Train the XGBoost models and save the feature importances
    models = train_xgboost_model(X_train, y_train, args.out_dir, feature_names)

    # Evaluate the model performance
    metrics = evaluate_model(models, X_test, y_test)

    # Persist the evaluation results
    results_file = os.path.join(args.out_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        # Write the table header
        f.write("Task | Accuracy | F1 | QWK | OrdMAE | NLL | Brier | AUROC | BrdECE\n")
        # Output the metrics for every task
        for metric in metrics:
            f.write(
                f"{metric.task} | {metric.accuracy:.4f} | {metric.f1_score:.4f} | "
                f"{metric.qwk:.4f} | {metric.ordmae:.4f} | {metric.nll:.4f} | "
                f"{metric.brier:.4f} | {metric.auroc:.4f} | {metric.brdece:.4f}\n"
            )

    print(f"Evaluation results saved to {results_file}")


if __name__ == "__main__":
    main()

