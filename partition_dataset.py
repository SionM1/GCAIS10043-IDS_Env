# partition_dataset.py

import pandas as pd
import numpy as np
import os

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
    for i in range(num_clients):
        start_idx = i * samples_per_client
        # Make sure last client takes any leftover rows
        end_idx = total_len if i == num_clients - 1 else (i+1) * samples_per_client

        client_df = df.iloc[start_idx:end_idx].copy()
        # 4) Save
        os.makedirs(output_dir, exist_ok=True)
        client_csv = os.path.join(output_dir, f"client_{i}.csv")
        client_df.to_csv(client_csv, header=False, index=False)
        print(f"Saved {len(client_df)} rows to {client_csv}")

if __name__ == "__main__":
    partition_iid(
        csv_path="car_hacking_dataset.csv",   # path to your 5-class data
        output_dir="client_data",             # folder to store splitted CSVs
        num_clients=10
    )
