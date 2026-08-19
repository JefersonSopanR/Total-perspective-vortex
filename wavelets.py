"""
wavelets.py

Bonus — Improve preprocessing using wavelet transform
========================================================
The mandatory pipeline gets its "power of the signal by frequency" via a
plain Fourier-based band-pass filter (preprocessing.py) feeding CSP's
variance-based feature. A Fourier transform tells you *which* frequencies
are present in a signal but throws away *when* they occurred; a wavelet
transform keeps both, at the cost of some frequency precision -- useful
because motor-imagery mu/beta desynchronization is a transient event tied
to the cue onset, not a constant background rhythm.

Two things live here:
  1. cwt_scalogram() -- a time-frequency visualization (a "scalogram"),
     the wavelet analogue of the PSD plots in preprocessing.py. Shows
     power across frequency AND time simultaneously for one channel.
  2. WaveletBandPower -- an sklearn-compatible transformer (an alternative feature-extraction approach to CSP) that extracts per-channel, per-frequency power
     using a continuous wavelet transform (Morlet wavelet) instead of a
     plain band-pass filter + variance.
"""

import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin

# Mu (~8-12 Hz) and beta (~13-30 Hz) rhythms, sampled every 2 Hz -- the
# same frequency range preprocessing.py's Fourier band-pass filter keeps,
# just resolved individually instead of lumped into one wide band.
DEFAULT_FREQS = np.arange(8.0, 31.0, 2.0)


def cwt_scalogram(channel_data: np.ndarray, sfreq: float, freqs: np.ndarray = DEFAULT_FREQS, wavelet: str = "morl"):
    """Continuous wavelet transform of one channel's time series.

    Returns (power, freqs, times) where power has shape (len(freqs),
    len(channel_data)) -- power[i, j] is the signal's power at freqs[i] and
    time times[j]. Feed this to a heatmap/imshow-style plot for a
    scalogram, the time-frequency analogue of a PSD plot.
    """
    scales = pywt.frequency2scale(wavelet, freqs / sfreq)
    coeffs, actual_freqs = pywt.cwt(channel_data, scales, wavelet, sampling_period=1.0 / sfreq)
    power = np.abs(coeffs) ** 2
    times = np.arange(channel_data.shape[-1]) / sfreq
    return power, actual_freqs, times


class WaveletBandPower(BaseEstimator, TransformerMixin):
    """Per-channel, per-frequency wavelet power as a feature vector --
    an alternative to CSP that skips spatial filtering and uses the
    "power of the signal by frequency and by channel" idea directly, via
    a continuous wavelet transform instead of a Fourier band-pass filter.

    Feature vector length = n_channels * len(freqs).

    Parameters
    ----------
    freqs : array-like, default 8..30 Hz in 2 Hz steps
        Frequencies (Hz) to compute wavelet power at.
    sfreq : float
        Sampling rate of the epochs in Hz (must match the data).
    wavelet : str, default "morl"
        PyWavelets continuous wavelet name.
    log : bool, default True
        Log-transform the power (same rationale as CSP's log-variance:
        keeps the feature distribution closer to Gaussian for LDA).
    """

    def __init__(self, freqs=None, sfreq: float = 160.0, wavelet: str = "morl", log: bool = True):
        self.freqs = freqs
        self.sfreq = sfreq
        self.wavelet = wavelet
        self.log = log

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        # Stateless -- nothing is learned from training data (unlike CSP's
        # spatial filters), but sklearn's Pipeline API expects a fit().
        self._freqs = np.asarray(self.freqs) if self.freqs is not None else DEFAULT_FREQS
        self._scales = pywt.frequency2scale(self.wavelet, self._freqs / self.sfreq)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_scales"):
            raise RuntimeError("WaveletBandPower is not fitted yet. Call fit before transform.")

        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("Expected X with shape (n_epochs, n_channels, n_times).")
        n_epochs, n_channels, n_times = X.shape
        features = np.empty((n_epochs, n_channels, len(self._freqs)))

        for i, epoch in enumerate(X):
            for ch in range(n_channels):
                coeffs, _ = pywt.cwt(epoch[ch], self._scales, self.wavelet, sampling_period=1.0 / self.sfreq)
                
                features[i, ch, :] = np.mean(np.abs(coeffs) ** 2, axis=1)  # avg power over time, per freq

        features = features.reshape(n_epochs, -1)  # (n_epochs, n_channels * n_freqs)
        if self.log:
            features = np.log(features + 1e-12)
        return features

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None, **fit_params) -> np.ndarray:
        return self.fit(X, y).transform(X)
	