import numpy as np

class GiniCriterion:
    def __init__(self):
        self.name = "gini"
    
    def calculate_impurity(self, data):
        """
        Calculate Gini impurity for a dataset.
        """
        if len(data) == 0:
            return 0
            
        # Get the label column
        label_column = data[:, -1]
        
        # Get unique labels and their counts
        _, counts = np.unique(label_column, return_counts=True)
        
        # Calculate probabilities
        probabilities = counts / counts.sum()
        
        # Calculate Gini impurity: 1 - Σ(p_i²)
        gini = 1 - np.sum(probabilities ** 2)
        
        return gini
    
    def calculate_overall_metric(self, data_below, data_above):
        """
        Calculate overall Gini impurity for a split. 
        """
        return calculate_overall_metric(data_below, data_above, self.calculate_impurity)

class EntropyCriterion:
    def __init__(self):
        self.name = "entropy"
    
    def calculate_impurity(self, data):
        """
        Calculate entropy for a dataset.
        """
        return calculate_entropy(data)
    
    def calculate_overall_metric(self, data_below, data_above):
        """
        Calculate overall entropy for a split.
        """
        return calculate_overall_metric(data_below, data_above, self.calculate_impurity)

class MSECriterion:
    def __init__(self):
        self.name = "mse"
    
    def calculate_impurity(self, data):
        """
        Calculate MSE for a dataset.
        """
        return calculate_mse(data)
    
    def calculate_overall_metric(self, data_below, data_above):
        """
        Calculate overall MSE for a split.
        """
        return calculate_overall_metric(data_below, data_above, self.calculate_impurity)

def calculate_mse(data):
    """
    Calculates the Mean Squared Error (MSE) of the target (label) values
    in the given dataset. This is used as a splitting criterion in 
    regression-based decision trees.

    Parameters:
    -----------
    data : np.ndarray
        A 2D NumPy array where each row represents a data instance,
        and the last column contains the target (label) values.

    Returns:
    --------
    mse : float
        The Mean Squared Error for the current dataset. If the dataset is empty,
        returns 0.
    """

    # Extract the label column (assumed to be the last column)
    actual_values = data[:, -1]

    # Edge case: if the dataset is empty, return MSE as 0
    if len(actual_values) == 0:
        mse = 0

    else:
        # Best constant prediction (minimizes squared error) is the mean
        prediction = np.mean(actual_values)

        # Compute MSE: average of squared differences from the mean
        mse = np.mean((actual_values - prediction) ** 2)

    return mse

def calculate_entropy(data):
    """
    Calculates the entropy of the class labels in the given dataset.

    Entropy is a measure of impurity or disorder in a dataset.
    In decision tree algorithms like ID3 and C4.5, entropy is used to
    evaluate the effectiveness of a potential split: lower entropy after
    a split means higher purity and better classification.

    Formula:
    --------
    Entropy(S) = -Σ (p_i * log₂(p_i)) for each class i

    Where:
    - p_i = probability of class i in the dataset (i.e., class frequency / total samples)

    Parameters:
    -----------
    - data (ndarray): A 2D NumPy array. Each row is a data sample.
                      The last column contains class labels.

    Returns:
    --------
    - entropy (float): The entropy value for the current dataset.
    """

    # Extract the last column, which contains the class labels
    label_column = data[:, -1]

    # Get the count of each unique class in the label column
    _, counts = np.unique(label_column, return_counts=True)

    # Convert counts to probabilities by dividing each count by the total number of samples
    probabilities = counts / counts.sum()

    # Apply the entropy formula: -Σ p_i * log₂(p_i)
    # NumPy handles element-wise multiplication and log2
    entropy = sum(probabilities * -np.log2(probabilities))

    # Return the entropy value
    return entropy

def calculate_overall_metric(data_below, data_above, metric_function):
    """
    Calculates the overall impurity or error of a split using a 
    weighted average of the metric values for the two partitions.
    
    This is used in decision trees to compare the effectiveness of 
    different potential splits.

    Parameters:
    -----------
    data_below : np.ndarray
        Subset of data where the split condition is True.
    
    data_above : np.ndarray
        Subset of data where the split condition is False.
    
    metric_function : function
        A function that computes a numeric score (impurity or error)
        for a given subset of data (e.g., entropy, Gini, MSE).

    Returns:
    --------
    overall_metric : float
        The weighted average impurity/error of the split.
    """

    # Total number of samples in both subsets
    n = len(data_below) + len(data_above)

    # Proportion of data in each subset
    p_data_below = len(data_below) / n
    p_data_above = len(data_above) / n

    # Weighted average of metric values from both subsets
    overall_metric = (
        p_data_below * metric_function(data_below) +
        p_data_above * metric_function(data_above)
    )

    return overall_metric
