# client.py

import sys
import flwr as fl
from flwr.client import NumPyClient
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

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
        # Set model weights to global parameters
        set_parameters(model, parameters)
        # Train locally for a few epochs
        train(model, X_train, y_train, epochs=5)
        # Return updated weights and number of training samples
        return get_parameters(model), len(X_train), {}

    def evaluate(self, parameters, config):
        # Update model to global parameters
        set_parameters(model, parameters)
        # Evaluate locally
        loss, accuracy = test(model, X_test, y_test)
        # Return evaluation stats
        return float(loss), len(X_test), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    print(f"Starting client {client_id}, CSV: {csv_path}")
    fl.client.start_client(server_address="localhost:8080", client=FlowerClient())




