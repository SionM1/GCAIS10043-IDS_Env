import pandas as pd
import numpy as np
import os
import csv  
import seaborn as sns

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from collections import defaultdict

def partition_iid(csv_path, output_dir, num_clients=5):
    """
    Loads `csv_path`, shuffles it, splits it into `num_clients` equal chunks.
    Saves each chunk to e.g. 'client_0.csv', 'client_1.csv', etc. in `output_dir`.
    """
    # 1) Load the dataset
    df = pd.read_csv(csv_path, header=None)

    # 2) Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # 3) Partition
    total_len = len(df)
    samples_per_client = total_len // num_clients

    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_clients):
        start_idx = i * samples_per_client
        end_idx = total_len if i == num_clients - 1 else (i+1) * samples_per_client

        client_df = df.iloc[start_idx:end_idx].copy()
        client_csv = os.path.join(output_dir, f"client_{i}.csv")
        client_df.to_csv(client_csv, header=False, index=False)
        print(f"Saved {len(client_df)} rows to {client_csv}")

def plot_class_distribution(data_dir, num_clients=5, plot_dir="visualization"):
    """
    Reads client_{i}.csv in `data_dir` and plots a heatmap of label counts
    across each client. The resulting figure is saved in `plot_dir`.
    """
    file_numbers = []
    flags = []
    counts = []

    for i in range(num_clients):
        file_name = f"client_{i}.csv"
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):
            print(f"{file_path} does not exist, skipping.")
            continue

        df = pd.read_csv(file_path, header=None)
        label_counts = df[df.columns[-1]].value_counts().sort_index()

        for label, cnt in label_counts.items():
            file_numbers.append(i)
            flags.append(label)
            counts.append(cnt)

    # Convert to DataFrame for plotting
    data = pd.DataFrame({"Client": file_numbers, "Class": flags, "Count": counts})
    # Pivot for heatmap
    heatmap_data = data.pivot(index="Class", columns="Client", values="Count").fillna(0)

    # Create the heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu")
    plt.title("Class Distribution across Clients")
    plt.xlabel("Client ID")
    plt.ylabel("Class Label")

    # Make sure the folder for the plot exists
    os.makedirs(plot_dir, exist_ok=True)
    
    # Save figure in `plot_dir`
    output_path = os.path.join(plot_dir, "class_distribution.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Heatmap has been saved to {output_path}")

if __name__ == "__main__":
    # 1) Partition the data => client_data/
    partition_iid(
        csv_path="car_hacking_dataset.csv",
        output_dir="client_data",
        num_clients=10
    )
    # 2) Generate a heatmap => visualization/class_distribution.png
    plot_class_distribution(data_dir="client_data", num_clients=10, plot_dir="visualization")
