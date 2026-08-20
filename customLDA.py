import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import matplotlib.pyplot as plt

class CustomLDA(BaseEstimator, ClassifierMixin):
    """A Custom Binary Linear Discriminant Analysis (LDA) Classifier built from scratch."""

    def __init__(self, shrinkage: float = 0.0, plot: bool = False, verbose: bool = False):
        # Default 0.0 uses the standard empirical-covariance LDA formulation.
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
        shrinkage = self.shrinkage  # Use the shrinkage value provided during initialization

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

        # 3. Calculate covariance for each class
        # rowvar=False ensures that each column is treated as a variable (feature)
        # and each row as an observation (sample)
        # It return the covariance of each feature with every other feature,
        # resulting in a square matrix of shape (n_features, n_features)
        # Since we have 6 features (features), cov_0 is a 6x6 grid
        cov_0 = np.cov(X_0, rowvar=False)
        cov_1 = np.cov(X_1, rowvar=False)
        n_0, n_1 = X_0.shape[0], X_1.shape[0]

        _log(f"cov_0 shape -> {cov_0.shape}")
        _log(f"cov_0 -> {cov_0}")
        _log(f"cov_1 shape -> {cov_1.shape}")
        _log(f"cov_1 -> {cov_1}")

        # 4. Calculate the pooled covariance matrix  
        # Pooled Covariance: it's the combination of the two covariances, giving more 
        # weight to the class that has more examples. That's why we multiply each 
        # covariance by its number of samples.
        pooled_cov = ((n_0 - 1) * cov_0 + (n_1 - 1) * cov_1) / (n_0 + n_1 - 2)
        _log(f"pooled_cov shape -> {pooled_cov.shape}")
        _log(f"pooled_cov -> {pooled_cov}")

        p = pooled_cov.shape[0] # Number of features (e.g., 768)
        # When features outnumber samples this is what actually protects against a
        # singular covariance matrix (protecting when Wavelet features are used, for example)
        if p > n_0 + n_1:
            shrinkage = 0.75  # Automatically dial up safety if features outnumber samples!
            _log(f"Shrinkage value -> {shrinkage}")
        _log(f"Shrinkage value -> {shrinkage}")
        # -------------------------------- Shrinkage Regularization -----------------------
        # When we use wavelets, we take the messy, highly correlated, crash-prone real data
        # the algorithm sees (too many features, too few samples), the math breaks and throws a "Singular Matrix" error
        # because a singular matrix is one that cannot be inverted.
        # This is a problem because we need to invert the covariance matrix to calculate the weights of the linear boundary.

        # inverting a matrix that only has numbers on the diagonal and 0s everywhere else is the easiest way to avoid a singular matrix.
        # this gurantees that the covariance matrix is always invertible

        # this creates a matrix where the diagonal is filled with the exact same number 
        # (the average variance of all your features), and every single off-diagonal number is exactly 0.
        target = np.identity(p) * np.trace(pooled_cov) / p

        # The resulting blended matrix is just structured enough to survive the np.linalg.inv() inversion,
        # but still keeps 20% of the real brainwave patterns so the classifier can actually do its job
        pooled_cov = (1 - shrinkage) * pooled_cov + shrinkage * target
        # ------------------------------------------------------------------------

        # 5. Invert the covariance matrix and calculate weights & intercept
        inv_cov = np.linalg.inv(pooled_cov)
        _log(f"inv_cov shape -> {inv_cov.shape}")

        # Seeing how a vector goes back after being transformed
        _log(f"Seeing how a vector goes back after being transformed")
        vector = self.mean_1_
        _log(f"Vector\n{vector}")
        tranformed_vector = np.dot(pooled_cov, vector)
        _log(f"Vector transfomed\n{tranformed_vector}")
        _log(f"Vector ditransfomed\n{np.dot(inv_cov, tranformed_vector)}")

        mean_diff = self.mean_1_ - self.mean_0_
        _log(f"mean_diff shape -> {mean_diff.shape}")
        _log(f"mean_diff -> {mean_diff}")

        # The geometric slope/weights of the dividing line
        # If a feature has a big mean_diff (strong signal) AND a small variance (smooth road), np.dot gives it a massive weight
        # If a feature has a big mean_diff but is incredibly noisy (mountain of static), np.dot squashes it down and gives it a tiny weight.
        # so with this we are finding the absolute smartest, cleanest, most noise-free combination of features to tell a left fist from a right fist!
        # self.coef_ = np.dot(inv_cov, mean_diff)
        # this calculates the pooled_cov inverse, and the np.dot(inv_cov, mean_diff)
        self.coef_ = np.linalg.solve(pooled_cov,mean_diff)
        _log(f"coef_ shape -> {self.coef_.shape}")
        _log(f"coef_ -> {self.coef_}")

        # The offset/intercept of the dividing line
        mean_sum = self.mean_1_ + self.mean_0_
        prior_0 = n_0 / (n_0 + n_1)
        prior_1 = n_1 / (n_0 + n_1)
        _log(f"prior_0 -> {prior_0}")
        _log(f"prior_1 -> {prior_1}")
        _log(f"plotting the shadows on the ruler")
        # Formula for the intercept is:
        # intercept = - Weights * (Mean of Class 1 + Mean of Class 0) / 2
        # The log(prior_1 / prior_0) term adjusts the intercept based on the relative sizes of the two classes.
        # if prior_1 > prior_0, the log term is positive, shifting the decision boundary towards Class 0,
        # making it easier to classify samples as Class 1.
        # With this we are finding the absolute smartest, cleanest, most noise-free combination of features
        # to distinguish a left fist from a right fist!
        weights = self.coef_
        self.intercept_ = -0.5 * np.dot(weights, mean_sum) + np.log(prior_1 / prior_0)
        _log(f"intercept_ -> {self.intercept_}")
        # ----------------- PLOTTING SECTION -----------------
        if self.plot:
            _log(f"plotting...")
            scores = np.dot(X, self.coef_) + self.intercept_
            scores_class0 = scores[y == 0]  
            scores_class1 = scores[y == 1]  

            plt.figure(figsize=(8, 5))
            plt.hist(scores_class0, bins=15, color='red', alpha=0.5, label='Class 0 (Left)')
            plt.hist(scores_class1, bins=15, color='blue', alpha=0.5, label='Class 1 (Right)')
            plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision Boundary (Score = 0)')

            plt.title(f"LDA 1D Projection (Shrinkage={shrinkage})")
            plt.xlabel("1D LDA Score")
            plt.ylabel("Number of Brainwave Epochs")
            plt.legend()
            
            # Use a unique timestamp or memory address so it NEVER overwrites
            #unique_id = int(time.time() * 1000)
            plt.savefig(f"lda_projection_.png") 
            plt.close()  # Cleanly close the figure from memory
        # ----------------------------------------------------
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
