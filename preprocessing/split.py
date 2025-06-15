import random

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