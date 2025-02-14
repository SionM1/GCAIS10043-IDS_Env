# server.py

import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
import tensorflow as tf
import numpy as np

from task import create_model, get_parameters

def weighted_average(metrics):
    """
     Combine client-reported accuracy into one global accuracy.
    Each elementin 'metrics' is a tuple: (num_examples, {"accuracy": acc}).
    """
    total_acc = 0.0
    total_samples = 0
    for (num_examples, metric_dict) in metrics:
        acc = metric_dict["accuracy"]
        total_acc += acc * num_examples
        total_samples += num_examples

    global_acc = total_acc / total_samples if total_samples > 0 else 0.0
    return {"accuracy": global_acc}

def main():
    # 1) Create an initial model
    init_model = create_model()
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(init_model))

    # 2) Define FedAvg strategy with the aggregator
    strategy = FedAvg(
        fraction_fit=1.0,       
        fraction_evaluate=1.0,  
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
        initial_parameters=initial_parameters,
        evaluate_metrics_aggregation_fn=weighted_average,  # aggregator function
    )

    print("Starting Flower server on localhost:8080...")
    fl.server.app.start_server(
        server_address="localhost:8080",
        config=ServerConfig(num_rounds=3),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
