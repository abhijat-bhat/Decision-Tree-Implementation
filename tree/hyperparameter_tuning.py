import numpy as np
from itertools import product
from sklearn.model_selection import KFold
from .decision_tree import DecisionTree
from evaluation.classification import calculate_accuracy
from evaluation.regression import calculate_r_squared
import pandas as pd

def tune_hyperparameters(X, y, ml_task, criterion_class, param_grid, n_splits=5):
    best_score = -np.inf if ml_task == "classification" else -np.inf # R2 can be negative, so this is fine
    best_params = {}
    best_model = None

    # Generate all combinations of hyperparameters
    keys = param_grid.keys()
    values = param_grid.values()
    hyperparameter_combinations = list(product(*values))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []

    for params in hyperparameter_combinations:
        current_params = dict(zip(keys, params))
        fold_scores = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            criterion_obj = criterion_class()
            model = DecisionTree(
                criterion=criterion_obj,
                max_depth=current_params.get('max_depth'),
                min_samples_split=current_params.get('min_samples_split'),
                min_samples_leaf=current_params.get('min_samples_leaf', 1) # Default to 1 if not in param_grid
            )

            model.fit(X_train, y_train)
            
            # Prepare validation DataFrame for custom evaluation functions
            val_df_for_eval = pd.concat([X_val, y_val.rename('label')], axis=1)

            if ml_task == "classification":
                score = calculate_accuracy(val_df_for_eval, model.tree)
            else: # regression
                score = calculate_r_squared(val_df_for_eval, model.tree)

            fold_scores.append(score)
        
        mean_score = np.mean(fold_scores)
        results.append({"params": current_params, "mean_score": mean_score})

        # Update best model based on task type
        if ml_task == "classification":
            if mean_score > best_score:
                best_score = mean_score
                best_params = current_params
                # Re-train the best model on the full training data for later use
                criterion_obj = criterion_class()
                best_model = DecisionTree(
                    criterion=criterion_obj,
                    max_depth=best_params.get('max_depth'),
                    min_samples_split=best_params.get('min_samples_split'),
                    min_samples_leaf=best_params.get('min_samples_leaf', 1)
                )
                best_model.fit(X, y)
        else: # regression
            if mean_score > best_score: # R2 score is higher for better models
                best_score = mean_score
                best_params = current_params
                # Re-train the best model on the full training data for later use
                criterion_obj = criterion_class()
                best_model = DecisionTree(
                    criterion=criterion_obj,
                    max_depth=best_params.get('max_depth'),
                    min_samples_split=best_params.get('min_samples_split'),
                    min_samples_leaf=best_params.get('min_samples_leaf', 1)
                )
                best_model.fit(X, y)

    return best_model, best_params, best_score, results 