import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier


def run_lr_attack(features, responses, test_size=0.2, seed=42):
    """Train and evaluate a Logistic Regression modeling attack."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, responses, test_size=test_size, random_state=seed
    )
    model = LogisticRegression(
        penalty='l2',
        solver='saga',
        max_iter=10_000,
        random_state=seed
    )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return acc, model


def run_xgb_attack(features, responses, test_size=0.2, seed=42):
    """Train and evaluate an XGBoost modeling attack."""
    X_train, X_test, y_train, y_test = train_test_split(
        features, responses, test_size=test_size, random_state=seed
    )
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=seed
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              verbose=False)
    acc = accuracy_score(y_test, model.predict(X_test))
    return acc, model


class MLPAttack(nn.Module):
    """Shallow MLP for PUF modeling attacks."""
    def __init__(self, input_dim, hidden_dims=(128, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class DNNAttack(nn.Module):
    """Deep neural network for high-capacity PUF modeling attacks."""
    def __init__(self, input_dim, hidden_dims=(256, 128, 64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(),
                       nn.Dropout(p=0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_nn_attack(model, features, responses,
                    epochs=100, batch_size=4096, lr=1e-3,
                    test_size=0.2, seed=42):
    """
    Train an MLP or DNN PUF modeling attack.

    Returns
    -------
    test_accuracy : float
        Prediction accuracy on clean held-out test CRPs.
    """
    torch.manual_seed(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        features, responses.astype(np.float32),
        test_size=test_size, random_state=seed
    )

    X_tr = torch.tensor(X_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    X_te = torch.tensor(X_test,  dtype=torch.float32)
    y_te = torch.tensor(y_test,  dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_tr, y_tr),
                        batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = (model(X_te.to(device)) > 0.5).cpu().numpy()
    acc = accuracy_score(y_te.numpy(), preds)
    return acc
