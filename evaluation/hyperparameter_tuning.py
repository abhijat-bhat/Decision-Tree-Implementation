import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tree.tree_builder import decision_tree_algorithm
from evaluation.classification import calculate_accuracy
from evaluation.regression import calculate_r_squared

def perform_grid_search(train_df, val_df, max_depth_values, min_samples_values, ml_task="regression"):
    """
    Performs grid search to tune hyperparameters of a decision tree for either regression or classification.

    Parameters:
    -----------
    train_df : pandas.DataFrame
        The training dataset. The last column is assumed to be the target variable (`label`).

    val_df : pandas.DataFrame
        The validation dataset. Same format as train_df.

    max_depth_values : list of int
        List of values to try for the `max_depth` hyperparameter (controls the maximum depth of the tree).

    min_samples_values : list of int
        List of values to try for the `min_samples` hyperparameter (minimum number of samples required to split).

    ml_task : str, optional (default="regression")
        Type of machine learning task: "regression" or "classification".

    Returns:
    --------
    grid_search_df : pandas.DataFrame
        DataFrame containing all hyperparameter combinations along with their performance (R² or Accuracy)
        on both the training and validation sets. Sorted by descending validation performance.
    """

    # Initialize a dictionary to collect results
    grid_search = {
        "max_depth": [],
        "min_samples": [],
    }

    # Add metric columns depending on task
    if ml_task == "regression":
        grid_search["r_squared_train"] = []
        grid_search["r_squared_val"] = []
    elif ml_task == "classification":
        grid_search["accuracy_train"] = []
        grid_search["accuracy_val"] = []
    else:
        raise ValueError("Invalid ml_task. Choose 'regression' or 'classification'.")

    # Loop over all combinations of hyperparameters
    for max_depth in max_depth_values:
        for min_samples in min_samples_values:

            # Train a decision tree with current hyperparameters
            tree = decision_tree_algorithm(
                df=train_df,
                ml_task=ml_task,
                max_depth=max_depth,
                min_samples=min_samples
            )

            # Predictions and metric calculation depend on task
            if ml_task == "regression":
                r_squared_train = calculate_r_squared(train_df, tree)
                r_squared_val = calculate_r_squared(val_df, tree)

                grid_search["r_squared_train"].append(r_squared_train)
                grid_search["r_squared_val"].append(r_squared_val)

            elif ml_task == "classification":
                accuracy_train = calculate_accuracy(train_df, tree)
                accuracy_val = calculate_accuracy(val_df, tree)

                grid_search["accuracy_train"].append(accuracy_train)
                grid_search["accuracy_val"].append(accuracy_val)

            # Store common hyperparameters
            grid_search["max_depth"].append(max_depth)
            grid_search["min_samples"].append(min_samples)

        # Optional: Print progress per max_depth value
        print(f"Progress: max_depth {max_depth} completed")

    # Convert the results into a DataFrame
    grid_search_df = pd.DataFrame(grid_search)

    # Sort by validation performance depending on task
    if ml_task == "regression":
        grid_search_df = grid_search_df.sort_values("r_squared_val", ascending=False)
    else:
        grid_search_df = grid_search_df.sort_values("accuracy_val", ascending=False)

    return grid_search_df

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
    None. Displays a heatmap of validation performance across hyperparameters.
    """

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
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        linewidths=0.5,
        linecolor="gray"
    )

    # Add labels and formatting
    plt.title(title, fontsize=14)
    plt.xlabel("min_samples")
    plt.ylabel("max_depth")
    plt.yticks(rotation=0)
    plt.show()