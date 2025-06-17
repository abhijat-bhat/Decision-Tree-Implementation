import pandas as pd
from tree.predict import predict_example

def calculate_r_squared(df, tree):
    """
    Computes the R² (coefficient of determination) for a regression tree model.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset that contains the features and the actual label values.

    tree : dict
        A trained regression decision tree (built using decision_tree_algorithm with ml_task='regression').

    Returns:
    --------
    float
        The R² score which measures the proportion of variance in the target variable
        that is predictable from the features.
    """
    
    # Step 1: Extract actual target values
    labels = df.label  # Assumes that the target column is named 'label'

    # Step 2: Compute mean of actual target values (used in SS_tot)
    mean = labels.mean()

    # Step 3: Make predictions using the regression tree for each row
    # `predict_example` should handle numeric output for regression
    predictions = df.apply(predict_example, args=(tree,), axis=1)

    # Step 4: Compute residual sum of squares (SS_res)
    # Measures prediction error (how far off predictions are from actual values)
    ss_res = sum((labels - predictions) ** 2)

    # Step 5: Compute total sum of squares (SS_tot)
    # Measures total variance in the actual values (how far actual values deviate from their mean)
    ss_tot = sum((labels - mean) ** 2)

    # Step 6: Compute R² score
    # 1 means perfect prediction; 0 means model performs no better than predicting the mean
    r_squared = 1 - ss_res / ss_tot

    return r_squared

def calculate_mean_absolute_error(df, tree):
    """
    Computes the Mean Absolute Error (MAE) for a regression tree model.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset that contains the features and the actual label values.

    tree : dict
        A trained regression decision tree.

    Returns:
    --------
    float
        The Mean Absolute Error (MAE).
    """

    labels = df.label
    predictions = df.apply(predict_example, args=(tree,), axis=1)

    mae = (abs(labels - predictions)).mean()

    return mae

def create_plot(df, tree, title, sort_by="index"):
    """
    Visualizes predicted vs actual values for regression using a line plot.

    Parameters:
    - df (DataFrame): The input DataFrame containing features and a 'label' column.
    - tree (dict): Trained decision tree used to make predictions.
    - title (str): Title of the plot.
    - sort_by (str): Column name to sort the data by for smoother plotting.
                     Use "index" to sort by DataFrame index.

    Returns:
    - None. Displays a line plot.
    """

    # Apply tree prediction on each row
    predictions = df.apply(predict_example, args=(tree,), axis=1)
    actual = df.label

    # Combine actual and predictions into a new DataFrame
    plot_df = pd.DataFrame({
        "actual": actual,
        "predictions": predictions
    })

    # Sort by index or a column
    if sort_by == "index":
        plot_df = plot_df.sort_index()
    elif sort_by in df.columns:
        plot_df = plot_df.sort_values(by=sort_by)

    # Plot the sorted results
    fig = plot_df.plot(
        figsize=(18, 5),
        title=title,
        xlabel="Data Index (sorted)",
        ylabel="Target Value"
    ).figure
    # plt.show() # Removed plt.show()
    return fig # Return the figure object