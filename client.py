# client.py

import sys
import flwr as fl
from flwr.client import NumPyClient
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_score, recall_score, f1_score


from task import create_model, train, test, get_parameters, set_parameters

if len(sys.argv) < 2:
    print("Usage: python client.py <client_id>")
    sys.exit()

client_id = int(sys.argv[1])  # e.g., 0..4
csv_path = f"client_data/client_{client_id}.csv"

# 1) Load local data
df = pd.read_csv(csv_path, header=None)
X = df.iloc[:, :-1].values  # all columns except last
y = df.iloc[:,  -1].values  # last column is label [0..4]

# 2) Simple local train/test split (e.g. 80% train, 20% test)
split_idx = int(0.8 * len(df))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# 3) Scale features
scaler = MinMaxScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# 4) Create local model
model = create_model()

# 5) Define Flower client
class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        # Return local model weights (only needed if server doesn't provide init)
        return get_parameters(model)

    def fit(self, parameters, config):
        # Update local model with global parameters
        set_parameters(model, parameters)
        # Train locally
        train(model, X_train, y_train, epochs=5)
        # Return updated weights
        return get_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config):
        # 1) Update local model with global parameters
        set_parameters(model, parameters)

        # 2) Evaluate on local test (loss & accuracy)
        loss, accuracy = test(model, X_test, y_test)

        # 3) Model predictions for extra metrics
        y_pred_probs = model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # 4) Calculate precision, recall, and F1 (weighted to handle class imbalance)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # Return all metrics for aggregator
        return float(loss), len(X_test), {
            "accuracy":  float(accuracy),
            "precision": float(prec),
            "recall":    float(rec),
            "f1_score":  float(f1),
        }

if __name__ == "__main__":
    print(f"Starting client {client_id}, CSV: {csv_path}")
    fl.client.start_client(server_address="localhost:8080", client=FlowerClient())



