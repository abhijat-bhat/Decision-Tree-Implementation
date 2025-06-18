import numpy as np
from tree.criteria import calculate_entropy, calculate_mse, calculate_overall_metric


def get_potential_splits(data):
    """
    Finds all potential split points for each feature (column) in the dataset.
 
    Parameters:
    - data (ndarray): A 2D NumPy array where the last column is the label 
                      and the rest are features (both categorical or continuous).

    Returns:
    - potential_splits (dict): A dictionary where each key is a column index (feature index), 
                               and the value is an array of unique values in that column.
                               These unique values represent all possible candidates for splitting.
    """
    
    potential_splits = {}
    _, n_columns = data.shape  # Get the number of columns in the dataset
                               # _ (underscore) is used to ignore the number of rows

    # Iterate over all columns except the last one (which is the label column)
    for column_index in range(n_columns - 1):
        values = data[:, column_index]        # Extract all values from the current feature column
        unique_values = np.unique(values)     # Get the unique values from that column
        
        # Store unique values as potential splits for the current column
        potential_splits[column_index] = unique_values

    return potential_splits

def determine_best_split(data, potential_splits, feature_types, criterion_obj):
    """
    Determines the best feature and value to split the dataset on, 
    by minimizing an overall impurity/error metric.

    This function supports both classification and regression tasks.

    Parameters:
    -----------
    data : np.ndarray
        The complete dataset (features + label) at the current node.
    
    potential_splits : dict
        A dictionary where keys are column indices and values are lists
        of values to try splitting on.
    
    feature_types : list of str
        A list indicating the type ('categorical' or 'continuous') of each feature column.

    criterion_obj : object
        The criterion object (e.g., GiniCriterion, EntropyCriterion, MSECriterion) to use for impurity calculation.

    Returns:
    --------
    best_split_column : int
        Index of the feature/column to split on.
    
    best_split_value : any
        The threshold or category value to use for the split.
    """

    first_iteration = True

    # Iterate over all features and their potential split values
    for column_index in potential_splits:
        for value in potential_splits[column_index]:

            # Split the data based on the current column and value
            data_below, data_above = split_data(
                data,
                split_column=column_index,
                split_value=value,
                feature_types=feature_types
            )

            # Use the provided criterion_obj for impurity calculation
            current_overall_metric = criterion_obj.calculate_overall_metric(data_below, data_above)

            # Update the best split if this is the first iteration,
            # or if the current split has a lower impurity/error
            if first_iteration or current_overall_metric <= best_overall_metric:
                first_iteration = False
                best_overall_metric = current_overall_metric
                best_split_column = column_index
                best_split_value = value

    return best_split_column, best_split_value, best_overall_metric

def split_data(data, split_column, split_value, feature_types):
    """
    Splits the dataset into two subsets based on the split column and split value.
    The method of splitting depends on whether the feature is continuous or categorical.

    Parameters:
    -----------
    data : numpy.ndarray
        The dataset to be split, where the last column is assumed to be the label.

    split_column : int
        The index of the column on which to split the data.

    split_value : float or str
        The value used to split the data. For continuous features, it's a numeric threshold.
        For categorical features, it's one of the discrete class values.

    feature_types : list of str
        A list indicating the type ('categorical' or 'continuous') of each feature column
        in the dataset (excluding the label column). This list should be computed separately
        using a helper function like `determine_type_of_feature()`.

    Returns:
    --------
    data_below : numpy.ndarray
        Subset of the dataset where the split condition is satisfied.

    data_above : numpy.ndarray
        Subset of the dataset where the split condition is not satisfied.
    """

    
    # Extract the values of the split column across all rows
    split_column_values = data[:, split_column]

    # Identify the type of the feature from the global list (must be defined elsewhere)
    type_of_feature = feature_types[split_column]

    # If the feature is continuous (e.g., sepal length, temperature)
    if type_of_feature == "continuous":
        # Split based on a numerical threshold
        data_below = data[split_column_values <= split_value]  # Values less than or equal to threshold
        data_above = data[split_column_values >  split_value]  # Values greater than threshold

    # If the feature is categorical (e.g., "red", "male", or class-encoded integers)
    else:
        # Split based on equality comparison
        data_below = data[split_column_values == split_value]  # Values equal to the split category
        data_above = data[split_column_values != split_value]  # Values not equal to the category
    
    return data_below, data_above
