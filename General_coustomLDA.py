import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import matplotlib.pyplot as plt

class CustomLDA(BaseEstimator, ClassifierMixin):
    """A Custom Binary Linear Discriminant Analysis (LDA) Classifier built from scratch."""

    def __init__(self, shrinkage: float = 0.0, plot: bool = False, verbose: bool = False):
        # Default 0.0 means "behave exactly like real LDA" -- no regularization
        # unless the data genuinely needs it. (When wavelet is used, shrinkage is often necessary.)
        self.shrinkage = shrinkage
        self.plot = plot
        self.verbose = verbose

    def fit(self, X, y):
        """Trains the LDA by finding class means and the pooled covariance matrix."""
        def _log(message):
            if self.verbose:
                print(message)

        _log("*" * 20)
        _log(f"Fitting CustomLDA with shrinkage={self.shrinkage}")
        # 1. Scikit-learn validation to ensure inputs are clean NumPy arrays
        X, y = check_X_y(X, y)
        _log(f" X shape -> {X.shape}")
        _log(f" y shape -> {y.shape}")
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("This custom LDA implementation is designed for binary classification (2 classes).")

        # Separate the data into Class 0 and Class 1 arrays
        X_0 = X[y == self.classes_[0]]
        X_1 = X[y == self.classes_[1]]
        _log(f"X_0 shape -> {X_0.shape}")
        _log(f"X_1 shape -> {X_1.shape}")

        # 2. Calculate the mean center of each class
        self.mean_0_ = np.mean(X_0, axis=0)
        self.mean_1_ = np.mean(X_1, axis=0)

        _log(f"mean_0_ shape -> {self.mean_0_.shape}")
        _log(f"mean_1_ shape -> {self.mean_1_.shape}")

        n_0, n_1 = X_0.shape[0], X_1.shape[0]


        mean_diff = self.mean_1_ - self.mean_0_
        _log(f"mean_diff shape -> {mean_diff.shape}")
        _log(f"mean_diff -> {mean_diff}")

        # 1. Center the data by subtracting the class mean from EVERY individual brainwave trial
        diff_X0 = X_0 - self.mean_0_  
        diff_X1 = X_1 - self.mean_1_  

        # 2. Calculate the scatter for each class using matrix multiplication (Dot Product)
        S_W_0 = np.dot(diff_X0.T, diff_X0)  
        S_W_1 = np.dot(diff_X1.T, diff_X1)  

        # 3. Add them together to get the total Within-Class Scatter
        S_W = S_W_0 + S_W_1
        inv_S_W = np.linalg.inv(S_W)

        # 1. Find the global center of ALL brainwaves combined
        mean_overall = np.mean(X, axis=0)

        # 2. Find the vector pointing from the global center to each class center
        diff_0 = self.mean_0_ - mean_overall
        diff_1 = self.mean_1_ - mean_overall

        # 3. Use the Outer Product to turn those 1D vectors into 768x768 grids
        S_B_0 = n_0 * np.outer(diff_0, diff_0)
        S_B_1 = n_1 * np.outer(diff_1, diff_1)

        # 4. Add them together
        S_B = S_B_0 + S_B_1

        # 3. SOLVE THE EIGENVALUE PROBLEM
        # We want the eigenvectors of (S_W^-1 dot S_B)
        target_matrix = np.dot(inv_S_W, S_B)

        # np.linalg.eig calculates the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(target_matrix)

        # 4. SELECT THE BEST VECTOR
        # The best separating line is the eigenvector with the largest eigenvalue.
        # We sort them in descending order and grab the top one.
        sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
        best_eigenvector = eigenvectors[:, sorted_indices[0]]

        # Note: Eigenvectors can come out with complex numbers (like 0.5 + 0.j) due to minor float errors.
        # We cast it to real numbers just to be safe.
        self.coef_ = np.real(best_eigenvector)




        mean_sum = self.mean_1_ + self.mean_0_
        prior_0 = n_0 / (n_0 + n_1)
        prior_1 = n_1 / (n_0 + n_1)
        weights = self.coef_
        self.intercept_ = -0.5 * np.dot(weights, mean_sum) + np.log(prior_1 / prior_0)
        _log(f"intercept_ -> {self.intercept_}")
        return self

    def predict(self, X):
        """Predicts the class of unseen data based on the linear boundary."""
        # Ensure the model was trained before predicting
        def _log(message):
            if self.verbose:
                print(message)
        _log(f"#" * 20)
        _log(f"Predicting with CustomLDA, shrinkage={self.shrinkage}")
        check_is_fitted(self)
        X = check_array(X)
        
        # Calculate the decision score for each sample: (X * weights) + intercept
        scores = np.dot(X, self.coef_) + self.intercept_
        
        # Threshold at 0: if score > 0, choose Class 1; else choose Class 0
        predictions = np.where(scores > 0, self.classes_[1], self.classes_[0])
        return predictions
