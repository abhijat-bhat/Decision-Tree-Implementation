import numpy as np

def create_leaf(data, ml_task):
    """
    Create a leaf node (prediction) for a decision tree, 
    depending on the type of machine learning task.

    Parameters:
    -----------
    data : np.ndarray
        The dataset at the current node. Each row is an example.
        The last column is assumed to be the label (target variable).

    ml_task : str
        Type of task. Accepts "regression" or "classification".

    Returns:
    --------
    leaf : float or object
        - For regression: returns the mean of the target values.
        - For classification: returns the most frequent class label.
    """
    
    # Extract the label column (assumed to be the last column)
    label_column = data[:, -1]

    # If the task is regression
    if ml_task == "regression":
        # Use the mean of the target values as the prediction
        # because minimizing Mean Squared Error leads to the mean
        leaf = np.mean(label_column)

    # If the task is classification
    else:
        # Count occurrences of each unique class
        unique_classes, counts_unique_classes = np.unique(label_column, return_counts=True)
        
        # Select the class with the maximum count (majority class)
        index = counts_unique_classes.argmax()
        leaf = unique_classes[index]
    
    return leaf

def check_purity(data):
    """
    Checks whether all examples in the dataset belong to the same class (i.e., pure node).
    
    Parameters:
    - data (ndarray): A 2D NumPy array where the last column contains class labels.

    Returns:
    - True if all labels in the dataset are the same (pure node), False otherwise.
    """

    # Extract the last column of the dataset, which contains the class labels
    label_column = data[:, -1]

    # Find the unique class labels present in the current subset of data
    unique_classes = np.unique(label_column)

    # If there is only one unique class, then the node is pure
    if len(unique_classes) == 1:
        return True
    else:
        return False
    