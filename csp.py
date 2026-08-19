"""
csp.py

Part V.1.3 — Implementation
============================
A from-scratch implementation of Common Spatial Patterns (CSP), the
dimensionality-reduction algorithm this project asks us to implement
ourselves. It is written as a scikit-learn compatible transformer
(BaseEstimator + TransformerMixin) so it can be dropped straight into an
sklearn Pipeline alongside any classifier, and used with
cross_val_score/GridSearchCV like any other transformer.

Math summary
------------
Given epoched EEG X of shape (n_epochs, n_channels, n_times) and binary
labels y, CSP looks for a projection matrix W (the "spatial filters") such
that the variance of the projected signal Z = W^T X is maximal for one
class and minimal for the other, simultaneously for every filter.

This is solved as a generalized eigenvalue problem on the two classes'
spatial covariance matrices:

    C1 w = lambda (C1 + C2) w

Eigenvectors with eigenvalues near 1 explain most of class 1's variance and
least of class 2's; eigenvectors with eigenvalues near 0 do the opposite.
Keeping a few eigenvectors from both ends of the spectrum gives the most
discriminative spatial filters. The transformed signal's log-variance
along the time axis is used as the feature fed to the classifier — this is
the standard CSP feature (and what makes the features roughly Gaussian,
which downstream linear classifiers like LDA rely on).
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from data import load_experiment_data
import matplotlib.pyplot as plt

# BaseEstimator, TransformerMixin are base classes from scikit-learn
# that provide a standard interface for building custom transformers and estimators.
# so this CSP class behave like a proper scikit-learn transformer,
# so it works seamlessly with the rest of the scikit-learn ecosystem
class CSP(BaseEstimator, TransformerMixin):
    """Common Spatial Patterns for binary EEG classification.

    Parameters
    ----------
    n_components : int, default=6
        Number of spatial filters to keep. Must be even: half are taken
        from the top of the eigenvalue spectrum (most class-1 variance)
        and half from the bottom (most class-2 variance).
    reg : float, default=1e-6
        Small ridge term added to the covariance matrices' diagonal before
        the eigendecomposition, for numerical stability when the number of
        time samples is close to the number of channels (near-singular
        covariance estimates).
    log : bool, default=True
        If True, features are log-variance (standard CSP). If False,
        features are the raw (normalized) variance.
    """

    def __init__(self, n_components: int = 6, reg: float = 1e-6, log: bool = True):
        self.n_components = n_components
        self.reg = reg
        self.log = log

    @staticmethod
    def _epoch_covariance(epoch: np.ndarray, reg: float) -> np.ndarray:
        """Spatial covariance of a single (n_channels, n_times) epoch,
        normalized by its trace so that epochs of different raw amplitude
        contribute comparably (standard CSP practice)."""
        cov = np.cov(epoch)
        # add a small ridge term to the diagonal to prevent singularity
        cov += reg * np.eye(cov.shape[0])
        trace = np.trace(cov)
        # Normalize each covariance matrix by its trace so that
        # CSP focuses on relative spatial variance rather than total signal energy.
        if trace > 0:
            cov = cov / trace
        return cov

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 3:
            raise ValueError(
                "X must have shape (n_epochs, n_channels, n_times)."
            )

        if y.ndim != 1:
            raise ValueError(
                "y must have shape (n_epochs,)."
            )

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of epochs."
            )

        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError(
                f"CSP as implemented here handles exactly 2 classes, got {classes.shape[0]}: {classes}. "
                "For >2 classes, train one-vs-rest CSP+classifier pairs."
            )
        if self.n_components % 2 != 0:
            raise ValueError("n_components must be even (split between the two classes).")

        self.classes_ = classes

        # Average, trace-normalized covariance matrix per class.
        cov_class = []
        for c in classes:
            epochs_c = X[y == c]
            covs = np.stack([self._epoch_covariance(e, self.reg) for e in epochs_c])
            cov_class.append(covs.mean(axis=0))
        c1, c2 = cov_class
        c_sum = c1 + c2

        # Generalized eigenvalue problem: c1 w = lambda (c1 + c2) w
        eigvals, eigvecs = eigh(c1, c_sum)

        # eigvecs columns are sorted ascending by eigval; take n/2 from each end.
        n_pairs = self.n_components // 2
        idx = np.concatenate([np.arange(n_pairs), np.arange(eigvecs.shape[1] - n_pairs, eigvecs.shape[1])])
        filters = eigvecs[:, idx].T  # shape (n_components, n_channels)

        self.filters_ = filters
        self.eigenvalues_ = eigvals[idx]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        if not hasattr(self, "filters_"):
            raise RuntimeError("CSP instance is not fitted yet. Call fit before transform.")

        # Project every epoch: (n_components, n_channels) @ (n_channels, n_times)
        projected = np.stack([self.filters_ @ epoch for epoch in X])  # (n_epochs, n_components, n_times)
        variances = projected.var(axis=2)  # (n_epochs, n_components)

        # Normalize each epoch by total CSP variance so that the features
        # represent relative component power rather than absolute signal energy.
        variances = variances / variances.sum(axis=1, keepdims=True)
        # log of 0 is undefined, so we add a small constant to avoid log(0)
        # log of a number between 0 and 1 is negative, the variance is normalized, so the log will be negative
        # a number closer to 0 will have a larger negative log value, and a number closer to 1 will have a smaller negative log value
        # Log-variance often reduces skewness and gives features a distribution
        # that is more suitable for classifiers such as LDA.
        if self.log:
            return np.log(variances + 1e-12)
        return variances

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, **fit_params) -> np.ndarray:
        return self.fit(X, y).transform(X)
