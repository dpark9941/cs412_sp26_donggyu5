
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import math

class RandomForestClassifier:
    def __init__(self, num_estimators, random_state=0, tree_max_depth=5):
        self.num_estimators = num_estimators
        self.random_state = random_state
        self.tree_max_depth = tree_max_depth
        self.models = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        self.models = []
        for i in range(self.num_estimators):
            rng = np.random.RandomState(self.random_state + i)
            # TODO: Implement bootstrap sampling and model training
        return self

    def predict(self, X):
        n_samples = X.shape[0]
        n_models = len(self.models)
        # TODO: Implement this method
        return np.array([])

class AdaBoostClassifier:
    """AdaBoost SAMME (multi-class) using DecisionTreeClassifier base learners."""
    def __init__(self, num_estimators, random_state=0):
        self.num_estimators = num_estimators
        self.random_state = random_state
        self.models = []
        self.alphas = []
        self.K = 0

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        classes = np.unique(y)
        self.K = len(classes)

        # initializations
        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples
        self.models = []
        self.alphas = []
        
        # TODO: Implement the rest of this method
        return self
        
    def predict(self, X):
        X = np.asarray(X)
        if not self.models:
            raise ValueError("AdaBoostClassifier is not fit yet.")
        # TODO: Implement this method, returning the predictions as a numpy array
        
        return None # 



if __name__ == "__main__":
    # Load the digits dataset
    print("Loading Digits dataset...")
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    
    # Split into train and test sets
    n_samples = X.shape[0]
    n_train = int(0.8 * n_samples)
    
    # Simple train-test split
    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Dataset: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print()
    
    # Test RandomForestClassifier
    print("=" * 50)
    print("Testing RandomForestClassifier")
    print("=" * 50)
    rf = RandomForestClassifier(num_estimators=10, random_state=0, tree_max_depth=5)
    rf.fit(X_train, y_train)
    rf_predictions = rf.predict(X_test)
    
    if len(rf_predictions) > 0:
        rf_accuracy = np.mean(rf_predictions == y_test)
        print(f"Number of models trained: {len(rf.models)}")
        print(f"Test Accuracy: {rf_accuracy:.4f}")
        if rf_accuracy > 0.79:
            print("RandomForestClassifier passed the accuracy threshold!")
        else:
            print("RandomForestClassifier did not pass the accuracy threshold, expected > 0.79")
    else:
        print("predict() method not yet implemented (returns empty array)")
    print()
    
    # Test AdaBoostClassifier
    print("=" * 50)
    print("Testing AdaBoostClassifier")
    print("=" * 50)
    ada = AdaBoostClassifier(num_estimators=50, random_state=0)
    ada.fit(X_train, y_train)
    ada_predictions = ada.predict(X_test)
    
    if len(ada_predictions) > 0:
        ada_accuracy = np.mean(ada_predictions == y_test)
        print(f"Number of models trained: {len(ada.models)}")
        print(f"Number of alphas: {len(ada.alphas)}")
        print(f"Test Accuracy: {ada_accuracy:.4f}")
        if ada_accuracy > 0.78:
            print("AdaBoostClassifier passed the accuracy threshold!")
        else:
            print("AdaBoostClassifier did not pass the accuracy threshold, expected > 0.78")
    else:
        print("predict() method not yet implemented (returns empty array)")
    print()
    
    print("Testing complete!")
