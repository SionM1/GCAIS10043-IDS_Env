# task.py

import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter

def create_model():
    """
    Creates a Keras model with two hidden layers (16 units each),
    and a final layer of 5 units (softmax) for multi-class classification.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(10,)),       # Adjust if you have a different # of features
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(5,  activation="softmax"),  # five-class output
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",   # For integer labels [0..4]
        metrics=["accuracy"]
    )
    return model

def compute_class_weights(y_train):
    """
    Returns a dictionary {class_index: weight} for Keras 'class_weight' argument,
    based on inverse frequency of each class in y_train.
    """
    counts = Counter(y_train)
    total_samples = len(y_train)
    num_classes = len(counts)  # 5
    class_weight = {}
    for cls, count in counts.items():
        class_weight[cls] = total_samples / (num_classes * count)
    return class_weight

def train(model, X_train, y_train, epochs=5):
    """
    Train the model using class weighting to handle imbalance.
    """
    class_weight = compute_class_weights(y_train)
    model.fit(
        X_train,
        y_train,
        batch_size=256,
        epochs=epochs,
        verbose=2, # 0 = silent, 1 = progress bar, 2 = one line per epoch
        class_weight=class_weight
    )
def test(model, X_test, y_test):
    """
    Evaluate the model, returns (loss, accuracy).
    """
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    return loss, acc

def get_parameters(model):
    """
    Retrieve model weights for Flower.
    """
    return [w.numpy() for w in model.weights]

def set_parameters(model, parameters):
    """
    Assign model weights from Flower updates.
    """
    for weight, param in zip(model.weights, parameters):
        weight.assign(param)