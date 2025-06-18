def determine_type_of_feature(df):
    """
    Determines whether each feature in the given DataFrame is 'categorical' or 'continuous'.
    
    Heuristics used:
    ----------------
    - If the feature has string-type values, it is classified as 'categorical'.
    - If the number of unique values in the feature is small (≤ 15), it is assumed to be categorical,
      even if the values are numeric.
    - Otherwise, the feature is treated as 'continuous'.

    Note:
    -----
    - The function skips the 'label' column, assuming it is the target variable.
    - The choice of threshold (15) is arbitrary and may be adjusted for different datasets.
    
    Returns:
    --------
    - feature_types : list of strings
        A list indicating the type ('categorical' or 'continuous') of each feature (excluding label).
    """
    
    feature_types = []  # Initialize an empty list to store the type of each feature
    
    n_unique_values_treshold = 7  # Threshold for deciding if a numeric feature is categorical
                                   # If a feature has ≤ 7 unique values, treat it as categorical
    
    # Iterate over each column in the DataFrame
    for feature in df.columns:

        # Skip the label column, since we do not want to classify the target variable
        if feature != "label":  
            
            # Extract all unique values from the current feature
            unique_values = df[feature].unique()
            
            # Select the first value from the unique values list
            # This will be used to check the data type
            example_value = unique_values[0]

            # Apply heuristic rules to determine the feature type:

            # Rule 1: If the first example value is a string (object), assume the whole column is categorical
            # (e.g., "red", "blue", "male", "female")
            if isinstance(example_value, str):
                feature_types.append("categorical")

            # Rule 2: If the number of unique values in the feature is less than or equal to the threshold (15),
            # we assume the feature is categorical.
            # This works in many real-world cases where categorical variables are numerically encoded
            # (e.g., 0: "Monday", 1: "Tuesday", ..., 6: "Sunday")
            elif len(unique_values) <= n_unique_values_treshold:
                feature_types.append("categorical")

            # Rule 3: If the above conditions are not satisfied, the feature is assumed to be continuous
            # (e.g., height, temperature, sepal length, etc.)
            else:
                feature_types.append("continuous")
    
    return feature_types  # Return the list of feature types (in order of appearance in the DataFrame)
