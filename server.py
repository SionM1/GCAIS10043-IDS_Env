# server.py

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
import tensorflow as tf
import numpy as np

from task import create_model, get_parameters

def main():
    # 1) Create an initial model
    init_model = create_model()
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(init_model))

    # 2) Define FedAvg strategy
    strategy = FedAvg(
        fraction_fit=1.0,       # fraction of clients to train each round
        fraction_evaluate=1.0,  # fraction of clients to evaluate
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,   # you have 5 total
        initial_parameters=initial_parameters,
        # no custom evaluate_fn by default, or you can define one
    )

    # 3) Start the Flower server on port 8080
    print("Starting Flower server on localhost:8080...")
    fl.server.app.start_server(
        server_address="localhost:8080",
        config=ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
