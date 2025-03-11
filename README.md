# Federated Learning Environment

A federated learning implementation using Flower framework for distributed machine learning with privacy preservation.

## Project Overview

This project implements a federated learning system where multiple clients can train a shared model while keeping their data private. The system uses a neural network for multi-class classification with class imbalance handling.

## Features

- Distributed model training across multiple clients
- Privacy-preserving learning (data never leaves client devices)
- Neural network with 2 hidden layers for multi-class classification
- Class imbalance handling through weighted training
- Comprehensive metric tracking (accuracy, precision, recall, F1-score)

## Technical Architecture

### Model Structure
- Input layer: 10 features
- Hidden layers: 2 layers with 16 units each (ReLU activation)
- Output layer: 5 classes (Softmax activation)
- Loss: Sparse Categorical Crossentropy
- Optimizer: Adam

### Components
- `task.py`: Core model architecture and training utilities
- `client.py`: Flower client implementation for federated learning
- `server.py`: Server-side aggregation and metric computation

## Getting Started

### Prerequisites
```bash
pip install flwr tensorflow numpy pandas
```

### Local Usage

1. Start the server:
```bash
python server.py
```

2. Launch clients (in separate terminals):
```bash
python client.py
```
### HPC Execution with Slurm

1. Submit the server job:
```bash
sbatch server.slurm
```

2. Submit multiple client jobs:
```bash
sbatch client.slurm


## Key Functions

- `create_model()`: Initializes the neural network architecture
- `compute_class_weights()`: Handles class imbalance in training data
- `train()`: Performs model training with class weighting
- `test()`: Evaluates model performance
- `get_parameters()`, `set_parameters()`: Handle model weight synchronization

## Metrics

The system tracks multiple performance metrics:
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1-Score (weighted)
### Contrubutions
Harry Jones - Model development https://github.com/jones-hdj/GCAIS10043-IDS
