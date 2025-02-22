import numpy as np
import pandas as pd
from tqdm import tqdm; tqdm.pandas()

def correct_label(row):
    dlc = row[2]
    flag = row[3+dlc]
    row[3+dlc] = np.nan
    row[11] = flag
    return row

# Read all CSVs and label them
df_norm = pd.read_csv("normal_dataset.csv", header=None, names=range(12)).progress_apply(correct_label, axis=1)
df_dos  = pd.read_csv("DoS_dataset.csv",    header=None, names=range(12)).progress_apply(correct_label, axis=1)
df_fuzz = pd.read_csv("Fuzzy_dataset.csv",  header=None, names=range(12)).progress_apply(correct_label, axis=1)
df_gear = pd.read_csv("gear_dataset.csv",   header=None, names=range(12)).progress_apply(correct_label, axis=1)
df_rpm  = pd.read_csv("RPM_dataset.csv",    header=None, names=range(12)).progress_apply(correct_label, axis=1)

# Map string flags -> numeric
df_norm[11] = df_norm[11].map(lambda x: 0)
df_dos[11]  = df_dos[11].map(lambda x: 0 if x=='R' else 1)
df_fuzz[11] = df_fuzz[11].map(lambda x: 0 if x=='R' else 2)
df_gear[11] = df_gear[11].map(lambda x: 0 if x=='R' else 3)
df_rpm[11]  = df_rpm[11].map(lambda x: 0 if x=='R' else 4)

# 3) Combine everything
df_conc = pd.concat([df_norm, df_dos, df_fuzz, df_gear, df_rpm])
df_conc = df_conc.drop(columns=0)
# Convert hex -> int
df_conc[[1,3,4,5,6,7,8,9,10]] = df_conc[[1,3,4,5,6,7,8,9,10]].fillna('0').map(lambda x: int(x,16))

# NEW: Undersampling the majority class 

# Split by label
df_0 = df_conc[ df_conc[11] == 0 ]
df_1 = df_conc[ df_conc[11] == 1 ]
df_2 = df_conc[ df_conc[11] == 2 ]
df_3 = df_conc[ df_conc[11] == 3 ]
df_4 = df_conc[ df_conc[11] == 4 ]

#Count minority classes
count_1 = len(df_1)
count_2 = len(df_2)
count_3 = len(df_3)
count_4 = len(df_4)

# pick the largest minority size:
largest_minority = max(count_1, count_2, count_3, count_4)

# Or pick total minority
# total_minority = count_1 + count_2 + count_3 + count_4


# Sample from df_0 so that class 0 has size ~largest_minority
df_0_undersampled = df_0.sample(n=largest_minority, random_state=42)
print(f"Original class-0 size: {len(df_0)} -> Undersampled to {len(df_0_undersampled)}")

# Re-combine
df_balanced = pd.concat([df_0_undersampled, df_1, df_2, df_3, df_4], ignore_index=True)

# Shuffle again 
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# save to CSV
with open("car_hacking_dataset.csv", "w") as f:
    for _, row in df_balanced.iterrows():
        f.write(",".join([str(x) for x in row]) + "\n")
