import numpy as np
import pandas as pd
from apuf_simulation import generate_apuf_crps, generate_xor_apuf_crps, generate_interpose_puf_crps
from noise_injection import inject_gaussian_noise
from ml_attacks import run_lr_attack, run_xgb_attack, MLPAttack, DNNAttack, train_nn_attack

NOISE_LEVELS  = [0.00, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
DATASET_SIZES = [200_000, 500_000, 1_000_000, 2_000_000]
N_STAGES      = 64
N_REPEATS_LR  = 5
N_REPEATS_DNN = 3

results = []

for n_crps in DATASET_SIZES:
    # --- Arbiter PUF ---
    feats, resps, _ = generate_apuf_crps(n_stages=N_STAGES, n_crps=n_crps)
    for sigma in NOISE_LEVELS:
        noisy = inject_gaussian_noise(feats, sigma) if sigma > 0 else feats

        acc_runs_lr  = [run_lr_attack(noisy, resps)[0] for _ in range(N_REPEATS_LR)]
        acc_runs_xgb = [run_xgb_attack(noisy, resps)[0] for _ in range(N_REPEATS_LR)]

        input_dim = feats.shape[1]
        acc_runs_mlp = [train_nn_attack(MLPAttack(input_dim), noisy, resps)
                        for _ in range(N_REPEATS_DNN)]
        acc_runs_dnn = [train_nn_attack(DNNAttack(input_dim), noisy, resps)
                        for _ in range(N_REPEATS_DNN)]

        results.append({
            'architecture': 'APUF',
            'n_crps'      : n_crps,
            'sigma'       : sigma,
            'lr_mean'     : np.mean(acc_runs_lr),
            'lr_std'      : np.std(acc_runs_lr),
            'xgb_mean'    : np.mean(acc_runs_xgb),
            'xgb_std'     : np.std(acc_runs_xgb),
            'mlp_mean'    : np.mean(acc_runs_mlp),
            'mlp_std'     : np.std(acc_runs_mlp),
            'dnn_mean'    : np.mean(acc_runs_dnn),
            'dnn_std'     : np.std(acc_runs_dnn),
        })
        print(f"APUF | n={n_crps} | sigma={sigma:.2f} | LR={np.mean(acc_runs_lr):.4f}")

df = pd.DataFrame(results)
df.to_csv('puf_dp_results.csv', index=False)
print(df.to_string())
