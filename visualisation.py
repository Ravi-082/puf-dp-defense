import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def measure_reliability(weights, challenges, sigma, n_trials=100, seed=42):
    """
    Measure PUF response reliability under repeated evaluations with noise.

    Parameters
    ----------
    weights    : ndarray  Device-specific delay weight vector.
    challenges : ndarray  Fixed challenge set to evaluate repeatedly.
    sigma      : float    Gaussian noise standard deviation.
    n_trials   : int      Number of repeated evaluations per challenge.
    seed       : int      Random seed.

    Returns
    -------
    reliability : float  Mean reliability across all challenges.
    """
    rng = np.random.default_rng(seed)
    n_crps = len(challenges)

    signed   = 1 - 2 * challenges
    features = np.cumprod(signed[:, ::-1], axis=1)[:, ::-1]
    features = np.hstack([features, np.ones((n_crps, 1))])
    ref_resp = (features @ weights > 0).astype(np.int8)

    flip_count = 0
    for _ in range(n_trials):
        noise       = rng.normal(0, sigma, size=features.shape)
        noisy_feats = features + noise
        noisy_resp  = (noisy_feats @ weights > 0).astype(np.int8)
        flip_count += np.sum(noisy_resp != ref_resp)

    reliability = 1.0 - flip_count / (n_crps * n_trials)
    return reliability


def plot_accuracy_vs_noise(df):
    """Plot ML attack accuracy vs noise level for all architectures and dataset sizes."""
    MODELS      = ['lr_mean', 'xgb_mean', 'mlp_mean', 'dnn_mean']
    LABELS      = ['LR', 'XGB', 'MLP', 'DNN']
    MARKERS     = ['o', 's', '^', 'D']
    ARCH        = ['APUF', 'XOR-APUF', 'Interpose PUF']
    SIZES       = [200_000, 500_000, 1_000_000, 2_000_000]
    SIZE_LABELS = ['200k', '500k', '1M', '2M']

    fig, axes = plt.subplots(len(ARCH), len(SIZES), figsize=(16, 10), sharey=True)

    for row, arch in enumerate(ARCH):
        for col, (sz, sz_lbl) in enumerate(zip(SIZES, SIZE_LABELS)):
            ax  = axes[row][col]
            sub = df[(df['architecture'] == arch) & (df['n_crps'] == sz)]
            sub = sub.sort_values('sigma')

            for model_col, label, marker in zip(MODELS, LABELS, MARKERS):
                ax.plot(sub['sigma'], sub[model_col],
                        marker=marker, label=label, linewidth=1.5)

            ax.axhline(0.5, color='grey', linestyle='--', linewidth=0.8, label='Random guess')
            ax.set_title(f'{arch} | N={sz_lbl}', fontsize=9)
            ax.set_xlabel(r'Noise level $\sigma$', fontsize=8)
            ax.set_ylabel('Attack accuracy', fontsize=8)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
            ax.set_ylim(0.40, 1.05)
            ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.7)
            if row == 0 and col == len(SIZES) - 1:
                ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig('figures/accuracy_vs_noise.pdf', dpi=300, bbox_inches='tight')
    plt.show()
