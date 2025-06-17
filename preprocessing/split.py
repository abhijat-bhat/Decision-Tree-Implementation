import random
import numpy as np
# from sklearn.model_selection import train_test_split as sk_train_test_split # Removed sklearn import
import pandas as pd

def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split the data into training and testing sets (custom implementation).
    
    Parameters:
    -----------
    X : array-like (pandas.DataFrame or numpy.ndarray)
        Features
    y : array-like (pandas.Series or numpy.ndarray)
        Target variable
    test_size : float, default=0.2
        Proportion of the dataset to include in the test split
    random_state : int, default=None
        Random state for reproducibility
        
    Returns:
    --------
    X_train, X_test, y_train, y_test : tuple
        Split data
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # Combine X and y into a single DataFrame for consistent splitting
    # This also helps maintain index alignment if X and y are pandas objects
    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        df = pd.concat([X, y], axis=1)
        original_index = df.index
    elif isinstance(X, np.ndarray) and isinstance(y, np.ndarray):
        df = np.hstack((X, y.reshape(-1, 1)))
        original_index = np.arange(len(df))
    else:
        raise TypeError("X and y must be both pandas objects or both numpy arrays.")

    n_samples = len(df)
    n_test = int(n_samples * test_size)
    
    # Generate shuffled indices
    shuffled_indices = list(range(n_samples))
    random.shuffle(shuffled_indices)

    test_indices = sorted(shuffled_indices[:n_test])
    train_indices = sorted(shuffled_indices[n_test:])

    if isinstance(X, pd.DataFrame) and isinstance(y, pd.Series):
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]
        y_train = y.iloc[train_indices]
        y_test = y.iloc[test_indices]
    else:
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]

    return X_train, X_test, y_train, y_test

def train_val_test_split(df, test_size, val_size, seed=None):
    """
    Custom function to split a DataFrame into training, validation, and testing sets.

    Parameters:
    - df (DataFrame): The input dataset to be split.
    - test_size (float or int): Proportion (0 < test_size < 1) or count of samples to use for testing.
    - val_size (float or int): Proportion (0 < val_size < 1) or count of samples to use for validation.
    - seed (int or None): Random seed for reproducibility. Default is None.

    Returns:
    - train_df (DataFrame): Subset used for training.
    - val_df (DataFrame): Subset used for validation.
    - test_df (DataFrame): Subset used for testing.
    """

    if seed is not None:
        random.seed(seed)  # Set seed for reproducibility

    total_len = len(df)

    # Convert proportions to absolute counts if floats are provided
    if isinstance(test_size, float):
        test_size = round(test_size * total_len)

    if isinstance(val_size, float):
        val_size = round(val_size * total_len)

    # Get all indices of the DataFrame
    indices = df.index.tolist()

    # Sample test indices
    test_indices = random.sample(indices, k=test_size)

    # Remaining indices after removing test set
    remaining_indices = list(set(indices) - set(test_indices))

    # Sample validation indices from remaining
    val_indices = random.sample(remaining_indices, k=val_size)

    # Remaining indices after removing validation set
    train_indices = list(set(remaining_indices) - set(val_indices))

    # Create subsets using the sampled indices
    test_df = df.loc[test_indices]
    val_df = df.loc[val_indices]
    train_df = df.loc[train_indices]

    return train_df, val_df, test_df