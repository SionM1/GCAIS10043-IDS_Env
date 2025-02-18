import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
import tensorflow as tf
import numpy as np

from task import create_model, get_parameters

def weighted_average(metrics):
    """
    Combine client-reported accuracy, precision, recall, and f1_score
    into a single global figure. Each element in 'metrics' is:
        (num_examples, {"accuracy": x, "precision": y, "recall": z, "f1_score": w})
    We'll do a weighted average by number of test examples.
    """

    total_acc  = 0.0
    total_prec = 0.0
    total_rec  = 0.0
    total_f1   = 0.0
    total_samples = 0

    for (num_examples, metric_dict) in metrics:
        total_acc  += metric_dict["accuracy"]  * num_examples
        total_prec += metric_dict["precision"] * num_examples
        total_rec  += metric_dict["recall"]    * num_examples
        total_f1   += metric_dict["f1_score"]  * num_examples
        total_samples += num_examples

    if total_samples == 0:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}

    return {
        "accuracy":  total_acc / total_samples,
        "precision": total_prec / total_samples,
        "recall":    total_rec  / total_samples,
        "f1_score":  total_f1   / total_samples,
    }

def main():
    # 1) Create an initial model
    init_model = create_model()
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(init_model))

    # 2) Define FedAvg strategy
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
