import random
import math
import numpy as np
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets

from homework2_q1 import RandomForestClassifier, AdaBoostClassifier

DECISION_TREE_MAX_DEPTH = 10
BAGGING_TREE_MAX_DEPTH = 5
BAGGING_NUM_ESTIMATORS = 25
BOOST_NUM_ESTIMATORS = 125

  

# Do not import any other libraries
def get_splits(n, k, y, seed):
    splits = None
    # Implement your code to construct the splits here
    # Do NOT change the return statement

    return splits

def my_cross_val(method, X, y, splits):
    errors = []
    # Implement your code to construct the list of errors here
    # Do NOT change the return statement
    
    return np.array(errors)

def get_model(method: str, random_state: int = 42):
    # Implement your code to return the model here.
    # Make sure to use the parameters specified in instructions.
    return None


if __name__ == "__main__":
    # Load the digits dataset
    print("Loading Digits dataset...")
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print()
    
    # Define methods to evaluate
    methods = [
        "DecisionTreeClassifier",
        "GaussianNB",
        "LogisticRegression",
        "RandomForestClassifier",
        "AdaBoostClassifier"
    ]
    
    # Perform k-fold cross-validation
    k = 5
    print(f"Performing {k}-fold cross-validation...")
    print("=" * 60)
    
    n = X.shape[0]
    splits = get_splits(n, k, y, seed=42)
    if len(splits) == k:
        print("Correct number of splits.")
    else:
        print("Incorrect number of splits.")
    if all(abs(len(s) - n // k) <= 1 for s in splits):
        print("Splits are balanced.")
    else:
        print("Splits are not balanced.")
    print()
    print("=" * 60)
    
    thresholds = [0.84, 0.83, 0.96, 0.80, 0.84]
    results = {}
    for method in methods:
        errors = my_cross_val(method, X, y, splits)
        accuracy = 1 - np.mean(errors)
        std = np.std(errors)
        results[method] = {"accuracy": accuracy, "std": std, "errors": errors}
        print(f"{method:30s} | Accuracy: {accuracy:.4f} (+/- {std:.4f})")
    
    print("=" * 60)
    print()
    
    # Find best method
    best_method = max(results, key=lambda m: results[m]["accuracy"])
    best_accuracy = results[best_method]["accuracy"]
    print(f"Best method: {best_method} with accuracy {best_accuracy:.4f}")
