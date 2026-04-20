import numpy as np


def inject_gaussian_noise(features, sigma, seed=None):
    """
    Inject additive Gaussian noise into parity-based feature vectors.

    The perturbed feature representation is:
        Phi_tilde(c) = Phi(c) + N(0, sigma^2)

    Parameters
    ----------
    features : ndarray, shape (n_crps, n_features)
        Clean parity-based feature vectors.
    sigma    : float
        Standard deviation of the Gaussian noise (noise magnitude).
    seed     : int or None
        Random seed for reproducibility.

    Returns
    -------
    noisy_features : ndarray, shape (n_crps, n_features)
        Noise-perturbed feature vectors.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=features.shape)
    return features + noise


def compute_dp_sigma(sensitivity, epsilon, delta=1e-5):
    """
    Compute the Gaussian mechanism noise parameter sigma from DP parameters.

    Uses the standard Gaussian mechanism formula:
        sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon

    Parameters
    ----------
    sensitivity : float  L2 sensitivity of the query function.
    epsilon     : float  Privacy budget (lower = stronger privacy).
    delta       : float  Failure probability parameter.

    Returns
    -------
    sigma : float  Required noise standard deviation.
    """
    return sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
