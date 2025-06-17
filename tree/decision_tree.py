import numpy as np
import pandas as pd
from .tree_builder import decision_tree_algorithm, FEATURE_IMPORTANCES
from .predict import predict_example

class DecisionTree:
    def __init__(self, criterion, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.ml_task = "classification" if self.criterion.name in ["gini", "entropy"] else "regression"
        self.feature_importances_ = None
        
    def fit(self, X, y):
        """
        Fit the decision tree to the data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The target vector
        """
        if isinstance(X, pd.DataFrame):
            X = X.copy()
        else:
            X = pd.DataFrame(X)
            
        if isinstance(y, pd.Series):
            y = y.copy()
        else:
            y = pd.Series(y)
            
        # Combine features and target
        df = pd.concat([X, y], axis=1)
        
        # Build the tree
        self.tree = decision_tree_algorithm(
            df=df,
            ml_task=self.ml_task,
            criterion_obj=self.criterion,
            counter=0,
            min_samples=self.min_samples_split,
            max_depth=self.max_depth
        )
        
        # Store feature importances
        self.feature_importances_ = FEATURE_IMPORTANCES
        
        return self
    
    def predict(self, X):
        """
        Make predictions for the input data.
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix to make predictions for
            
        Returns:
        --------
        numpy.ndarray
            The predicted values
        """
        if self.tree is None:
            raise ValueError("Tree has not been fitted yet. Call fit() first.")
            
        if isinstance(X, pd.DataFrame):
            X = X.copy()
        else:
            X = pd.DataFrame(X)
            
        predictions = []
        for _, row in X.iterrows():
            prediction = predict_example(row, self.tree)
            predictions.append(prediction)
            
        return np.array(predictions)
    
    def score(self, X, y):
        """
        Calculate the score (accuracy for classification, R² for regression).
        
        Parameters:
        -----------
        X : pandas.DataFrame or numpy.ndarray
            The feature matrix
        y : pandas.Series or numpy.ndarray
            The true target values
            
        Returns:
        --------
        float
            The score
        """
        predictions = self.predict(X)
        
        if self.ml_task == "classification":
            return np.mean(predictions == y)
        else:
            # R² score for regression
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            return 1 - (ss_res / ss_tot) 