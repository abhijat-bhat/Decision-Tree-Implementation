def predict_example(example, tree):
    # Extract the current question from the decision tree node (the only key in the dictionary)
    question = list(tree.keys())[0]
    
    # Split the question into its components: feature name, comparison operator, and threshold value
    feature_name, comparison_operator, value = question.split(" ")

    # -----------------------------------------------------
    # Determine the answer to the question based on feature type
    # -----------------------------------------------------
    
    # Case 1: Continuous feature (uses '<=' operator)
    if comparison_operator == "<=":
        # If the value of the feature in the input example is <= threshold, take the 'yes' branch
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        # Otherwise, take the 'no' branch
        else:
            answer = tree[question][1]
    
    # Case 2: Categorical feature (uses '=' operator)
    else:
        # If the feature value matches the split category, take the 'yes' branch
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        # Otherwise, take the 'no' branch
        else:
            answer = tree[question][1]

    # -----------------------------------------------------
    # Determine if the answer is a leaf node or another subtree
    # -----------------------------------------------------
    
    # Base Case: If the answer is not a dictionary, it is a class label (leaf node)
    if not isinstance(answer, dict):
        return answer
    
    # Recursive Case: If the answer is another dictionary (subtree), continue traversing
    else:
        residual_tree = answer
        return predict_example(example, residual_tree)