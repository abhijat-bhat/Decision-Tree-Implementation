from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tree.predict import predict_example
import matplotlib.pyplot as plt

def calculate_accuracy(df, tree):
    """
    Computes the classification accuracy of a decision tree model on a given DataFrame.

    Parameters:
    - df (DataFrame): The input DataFrame containing the dataset. It must include a column named 'label' 
                      representing the true class labels.
    - tree (dict): A trained decision tree model represented as a nested dictionary (from a custom implementation).

    Returns:
    - accuracy (float): The proportion of correctly classified instances, i.e., the accuracy score.
    """

    # Step 1: Apply the prediction function to each row of the DataFrame
    # 'predict_example' is assumed to be a user-defined function that takes a row and the decision tree
    # and returns the predicted class label for that row
    df["classification"] = df.apply(predict_example, args=(tree,), axis=1)

    # Step 2: Create a new Boolean column that is True if the prediction matches the actual label
    # This column is used to identify correctly classified examples
    df["classification_correct"] = df["classification"] == df["label"]

    # Step 3: Compute the mean of the Boolean column
    # Since True is treated as 1 and False as 0, the mean gives the fraction of correct predictions
    accuracy = df["classification_correct"].mean()

    # Step 4: Return the computed accuracy
    return accuracy

def plot_confusion_matrix(df, tree, title="Confusion Matrix"):
    """
    Plots the confusion matrix for a classification decision tree.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataset, must include a 'label' column with ground truth class labels.
    
    tree : dict
        The trained classification decision tree structure.
    
    title : str
        Title of the plot.

    Returns:
    --------
    None (displays a confusion matrix plot).
    """

    # Predict labels using the decision tree
    predictions = df.apply(predict_example, args=(tree,), axis=1)

    # Extract actual labels
    actual = df['label']

    # Compute confusion matrix
    cm = confusion_matrix(actual, predictions)

    # Display the confusion matrix with proper labels
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.grid(False)
    plt.show()