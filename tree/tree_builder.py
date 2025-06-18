from tree.helper import create_leaf, check_purity
from preprocessing.feature_types import determine_type_of_feature
from tree.splits import get_potential_splits, determine_best_split, split_data
import numpy as np

def decision_tree_algorithm(df, ml_task, criterion_obj, counter=0, min_samples=2, max_depth=5, COLUMN_HEADERS=None, FEATURE_TYPES=None, FEATURE_IMPORTANCES=None):
    """
    Recursively builds a decision tree for either classification or regression.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input data containing features and the target label as the last column.
    
    ml_task : str
        Specifies the type of machine learning task - "classification" or "regression".
    
    criterion_obj : object
        The criterion object used to calculate impurity and determine the best split.
    
    counter : int (default = 0)
        Keeps track of the current depth of the tree during recursion.

    min_samples : int (default = 2)
        The minimum number of samples required to make a further split.

    max_depth : int (default = 5)
        The maximum allowed depth of the decision tree.

    COLUMN_HEADERS : list (default = None)
        List of column headers for the input DataFrame.

    FEATURE_TYPES : dict (default = None)
        Dictionary mapping feature names to their types ("continuous" or "categorical").

    FEATURE_IMPORTANCES : dict (default = None)
        Dictionary mapping feature names to their importance scores.

    Returns:
    --------
    dict or value:
        A nested dictionary representing the decision tree if more splits are made,
        or a leaf value (class label or mean prediction) if a base case is reached.
    """

    # === Step 1: Data Preparation (Only once, at root level) ===
    if counter == 0:
        # Save column names and determine feature types globally
        COLUMN_HEADERS = df.columns
        FEATURE_TYPES = determine_type_of_feature(df)
        FEATURE_IMPORTANCES = {col: 0.0 for col in COLUMN_HEADERS[:-1]} # Initialize with 0 for all features
        # Convert DataFrame to NumPy array for faster processing
        data = df.values
    else:
        # In recursive calls, data is already in NumPy array format
        data = df

    # === Step 2: Base Cases for Stopping the Recursion ===
    # Stop recursion and return a leaf node if:
    # - Data is pure (all labels are the same)
    # - Number of rows is below minimum allowed
    # - Tree has reached the maximum depth
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        leaf = create_leaf(data, ml_task)  # Use mean (regression) or mode (classification)
        return leaf, FEATURE_IMPORTANCES

    # === Step 3: Recursive Case ===
    else:
        counter += 1  # Increment tree depth

        # Safety check: ensure required variables are not None in recursion
        if COLUMN_HEADERS is None or FEATURE_TYPES is None or FEATURE_IMPORTANCES is None:
            raise ValueError("COLUMN_HEADERS, FEATURE_TYPES, and FEATURE_IMPORTANCES must not be None in recursion.")

        # Calculate initial impurity before splitting
        initial_impurity = criterion_obj.calculate_impurity(data)

        # Find the best column and value to split the data
        potential_splits = get_potential_splits(data)
        split_column, split_value, overall_metric_after_split = determine_best_split(
            data, potential_splits, feature_types=FEATURE_TYPES, criterion_obj=criterion_obj)

        # Calculate impurity reduction and update feature importance
        impurity_reduction = initial_impurity - overall_metric_after_split
        FEATURE_IMPORTANCES[COLUMN_HEADERS[split_column]] += impurity_reduction

        # Divide the data based on the chosen split
        data_below, data_above = split_data(data, split_column, split_value, feature_types=FEATURE_TYPES)

        # If either side is empty after split, return a leaf (prevents invalid recursion)
        if len(data_below) == 0 or len(data_above) == 0:
            leaf = create_leaf(data, ml_task)
            return leaf, FEATURE_IMPORTANCES

        # === Step 4: Define the Question for the Current Node ===
        feature_name = COLUMN_HEADERS[split_column]           # Get feature name
        type_of_feature = FEATURE_TYPES[split_column]         # Check if categorical or continuous

        if type_of_feature == "continuous":
            # Format question for numeric/continuous features
            question = "{} <= {}".format(feature_name, split_value)
        else:
            # Format question for categorical features
            question = "{} = {}".format(feature_name, split_value)

        # Create a dictionary for this internal node in the tree
        sub_tree = {question: []}

        # === Step 5: Recur for Left and Right Subtrees ===
        # Recursively build the left (yes) and right (no) branches
        yes_answer, yes_imp_after = decision_tree_algorithm(
            data_below, ml_task, criterion_obj, counter, min_samples, max_depth,
            COLUMN_HEADERS=COLUMN_HEADERS, FEATURE_TYPES=FEATURE_TYPES, FEATURE_IMPORTANCES=FEATURE_IMPORTANCES
        )

        no_answer, no_imp_after = decision_tree_algorithm(
            data_above, ml_task, criterion_obj, counter, min_samples, max_depth,
            COLUMN_HEADERS=COLUMN_HEADERS, FEATURE_TYPES=FEATURE_TYPES, FEATURE_IMPORTANCES=FEATURE_IMPORTANCES
        )

        # === Step 6: Handle Case Where Both Branches Are Identical ===
        # This can happen if the split didn't add any new information.
        if yes_answer == no_answer:
            # Replace subtree with a single leaf node
            sub_tree = yes_answer
        else:
            # Append both branches to the current node
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)

        return sub_tree, FEATURE_IMPORTANCES

