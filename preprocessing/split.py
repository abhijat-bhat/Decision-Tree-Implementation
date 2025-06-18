import random
import numpy as np
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
