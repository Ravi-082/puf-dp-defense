# PUF Differential Privacy Defense

Source code for the thesis: *Differential Privacy-Inspired Gaussian Noise Injection as a Defense Against ML-Based Modeling Attacks on Physical Unclonable Functions*.

## Files

| File | Description |
|------|-------------|
| `apuf_simulation.py` | Arbiter PUF, XOR-APUF, and Interpose PUF simulation |
| `noise_injection.py` | Gaussian noise injection and DP sigma computation |
| `ml_attacks.py` | LR, XGBoost, MLP, and DNN attack models |
| `experiment_runner.py` | Automated experiment loop across all architectures and noise levels |
| `visualisation.py` | Accuracy vs noise plotting and reliability measurement |

## Setup

```bash
python3 -m venv puf_env
source puf_env/bin/activate   # Linux/macOS
# puf_env\Scripts\activate    # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

```bash
python experiment_runner.py
```
