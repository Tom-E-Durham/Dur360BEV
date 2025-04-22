"""
Generate a dataset split (train, test, val) and save as a .pkl file.
The range 0-1639 is split into train (80%), test (10%), and validation (10%).
"""

import pickle
import random

# Define the total range of indices
indices = list(range(1600))  # 0 to 1599
random.shuffle(indices)  # Shuffle to ensure randomness

# Define split ratios
train_size = int(0.9 * len(indices))  # 80%
test_size = int(0.1 * len(indices))   # 10%
# val_size = len(indices) - train_size - test_size  # Remaining 10%

# Split indices
train_indices = indices[:train_size]
test_indices = indices[train_size:train_size + test_size]
# val_indices = indices[train_size + test_size:]

# Store in a dictionary
data_splits = {
    "train_indices": train_indices,
    "test_indices": test_indices,
}

# Save to a .pkl file
with open("/media/tom/SSD1/DurLar-360/My_dataset/data/minidata/dataset_indices_random.pkl", "wb") as f:
    pickle.dump(data_splits, f)

print("Dataset indices saved to dataset_indices_random.pkl")