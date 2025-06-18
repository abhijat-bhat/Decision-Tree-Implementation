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
                max_depth=int(current_params.get('max_depth', 5)) if current_params.get('max_depth', 5) is not None else 5,
                min_samples_split=int(current_params.get('min_samples_split', 2)) if current_params.get('min_samples_split', 2) is not None else 2,
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
        result_row = dict(current_params)
        if ml_task == "classification":
            result_row["accuracy_val"] = mean_score
        else:
            result_row["r_squared_val"] = mean_score
        results.append(result_row)

        # Update best model based on task type
        if ml_task == "classification":
            if mean_score > best_score:
                best_score = mean_score
                best_params = current_params
                # Re-train the best model on the full training data for later use
                criterion_obj = criterion_class()
                best_model = DecisionTree(
                    criterion=criterion_obj,
                    max_depth=int(best_params.get('max_depth', 5)) if best_params.get('max_depth', 5) is not None else 5,
                    min_samples_split=int(best_params.get('min_samples_split', 2)) if best_params.get('min_samples_split', 2) is not None else 2,
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
                    max_depth=int(best_params.get('max_depth', 5)) if best_params.get('max_depth', 5) is not None else 5,
                    min_samples_split=int(best_params.get('min_samples_split', 2)) if best_params.get('min_samples_split', 2) is not None else 2,
                    min_samples_leaf=best_params.get('min_samples_leaf', 1)
                )
                best_model.fit(X, y)

    return best_model, best_params, best_score, results 

def plot_grid_search_heatmap(grid_search_df):
    """
    Visualizes grid search results as a heatmap for either regression (R² score)
    or classification (accuracy), based on the available metric in the DataFrame.

    Parameters:
    -----------
    grid_search_df : pandas.DataFrame
        DataFrame with hyperparameter tuning results. It must contain:
        - 'max_depth' (int): Values of max tree depth.
        - 'min_samples' (int): Minimum samples required to split.
        - Either:
            - 'r_squared_val' (float): Validation R² for regression.
            - OR 'accuracy_val' (float): Validation accuracy for classification.

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The matplotlib figure object for the heatmap.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Detect the metric type
    if "r_squared_val" in grid_search_df.columns:
        metric = "r_squared_val"
        title = "Grid Search Heatmap: R² Validation Score"
        fmt = ".3f"
        cmap = "YlGnBu"
    elif "accuracy_val" in grid_search_df.columns:
        metric = "accuracy_val"
        title = "Grid Search Heatmap: Classification Accuracy"
        fmt = ".2%"  # display accuracy as percentage
        cmap = "YlOrRd"
    else:
        raise ValueError("The DataFrame must contain either 'r_squared_val' or 'accuracy_val' column.")

    # Pivot DataFrame for heatmap format
    heatmap_data = grid_search_df.pivot(
        index="max_depth",
        columns="min_samples",
        values=metric
    )

    # Plot the heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.5,
        linecolor="gray",
        ax=ax
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("min_samples")
    ax.set_ylabel("max_depth")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    return fig 