import numpy as np

def generate_apuf_crps(n_stages=64, n_crps=2_000_000, seed=42):
    """
    Simulate an Arbiter PUF using the analytic linear delay model.

    Parameters
    ----------
    n_stages : int
        Number of PUF stages (challenge bit length).
    n_crps   : int
        Number of Challenge-Response Pairs to generate.
    seed     : int
        Random seed for reproducibility.

    Returns
    -------
    features : ndarray, shape (n_crps, n_stages + 1)
        Parity-based feature vectors Phi(c) in {-1, +1}.
    responses : ndarray, shape (n_crps,)
        Binary PUF responses r in {0, 1}.
    weights : ndarray, shape (n_stages + 1,)
        Device-specific delay weight vector w ~ N(0, 1).
    """
    rng = np.random.default_rng(seed)

    # Device-specific delay vector: w ~ N(0, 1)
    weights = rng.standard_normal(n_stages + 1)

    # Random binary challenges: c_i ~ Bernoulli(0.5)
    challenges = rng.integers(0, 2, size=(n_crps, n_stages))

    # Parity-based feature extraction: Phi_i = prod_{j=i}^{n} (-1)^{c_j}
    signed = 1 - 2 * challenges          # map {0,1} -> {+1,-1}
    features = np.cumprod(signed[:, ::-1], axis=1)[:, ::-1]

    # Append bias term
    features = np.hstack([features, np.ones((n_crps, 1))])

    # Delay difference: Delta = w^T * Phi(c)
    delay_diff = features @ weights

    # Binary response: r = 1 if Delta > 0, else 0
    responses = (delay_diff > 0).astype(np.int8)

    return features, responses, weights


def generate_xor_apuf_crps(n_stages=64, k=6, n_crps=2_000_000, seed=42):
    """
    Simulate a k-XOR Arbiter PUF.

    Parameters
    ----------
    n_stages : int   Number of stages per Arbiter PUF component.
    k        : int   Number of Arbiter PUFs to XOR-combine.
    n_crps   : int   Number of CRPs to generate.
    seed     : int   Random seed.

    Returns
    -------
    features  : ndarray  Parity feature vectors (same challenge for all APUFs).
    responses : ndarray  XOR-combined binary responses.
    """
    rng = np.random.default_rng(seed)

    # Shared challenges across all k APUFs
    challenges = rng.integers(0, 2, size=(n_crps, n_stages))
    signed = 1 - 2 * challenges
    features = np.cumprod(signed[:, ::-1], axis=1)[:, ::-1]
    features = np.hstack([features, np.ones((n_crps, 1))])

    # XOR responses from k independent APUFs
    responses = np.zeros(n_crps, dtype=np.int8)
    for _ in range(k):
        w_i = rng.standard_normal(n_stages + 1)
        delay_i = features @ w_i
        r_i = (delay_i > 0).astype(np.int8)
        responses = np.bitwise_xor(responses, r_i)

    return features, responses


def generate_interpose_puf_crps(n_stages=64, k_up=2, k_down=2,
                                 n_crps=2_000_000, interpose_bit=32,
                                 seed=42):
    """
    Simulate an Interpose PUF (IPUF).

    The upper XOR-APUF produces an intermediate bit f_u(c), which is
    inserted at position `interpose_bit` of the challenge vector before
    feeding into the lower XOR-APUF.

    Parameters
    ----------
    n_stages      : int  Number of stages.
    k_up          : int  XOR width of the upper APUF block.
    k_down        : int  XOR width of the lower APUF block.
    n_crps        : int  Number of CRPs.
    interpose_bit : int  Bit position for interposition.
    seed          : int  Random seed.

    Returns
    -------
    responses : ndarray  Final binary responses.
    """
    rng = np.random.default_rng(seed)

    challenges = rng.integers(0, 2, size=(n_crps, n_stages))

    def xor_apuf_response(chal, k, rng_local):
        signed = 1 - 2 * chal
        feats = np.cumprod(signed[:, ::-1], axis=1)[:, ::-1]
        feats = np.hstack([feats, np.ones((len(chal), 1))])
        resp = np.zeros(len(chal), dtype=np.int8)
        for _ in range(k):
            w = rng_local.standard_normal(n_stages + 1)
            resp ^= (feats @ w > 0).astype(np.int8)
        return resp

    # Step 1: Upper block generates intermediate response
    f_upper = xor_apuf_response(challenges, k_up, rng)

    # Step 2: Insert f_upper at interpose_bit position
    c_prime = np.hstack([
        challenges[:, :interpose_bit],
        f_upper.reshape(-1, 1),
        challenges[:, interpose_bit:]
    ])

    # Step 3: Lower block uses modified challenge (truncated to n_stages)
    c_lower = c_prime[:, :n_stages]
    responses = xor_apuf_response(c_lower, k_down, rng)

    return responses
