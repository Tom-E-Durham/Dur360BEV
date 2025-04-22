from fire import Fire
import pickle
from torchvision import transforms
import my_dataset
import torch
import numpy as np
from torch.utils.data import Subset
import sys
sys.path.append("..")


def random_splits(dataset, train_frac, val_frac, seed=42):
    torch.manual_seed(seed)  # For reproducibility
    # Generate a shuffled list of indices
    total_size = len(dataset)
    indices = torch.randperm(total_size).tolist()

    train_size = int(train_frac * total_size)
    val_size = int(val_frac * total_size)
    test_size = total_size - train_size - val_size

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, val_subset, test_subset, train_indices, val_indices, test_indices


def distance_splits(dataset):

    # define the test set range: arround 850 meters, 1640 pieces of data
    test_start_index = 11510
    test_end_index = test_start_index + 1640

    # generate indices
    all_indices = np.arange(len(dataset))
    test_indices = np.arange(test_start_index, test_end_index)
    train_indices = np.delete(all_indices, test_indices)

    # create subsets
    train_subset = Subset(dataset, train_indices)
    test_subset = Subset(dataset, test_indices)

    return train_subset, test_subset, train_indices, test_indices


def main(random_split=False):
    # load dataset
    root_dir = "../data/2024-03-12"
    dataset = my_dataset.Dur360(root_dir, transform=transforms.ToTensor())

    if random_split:
        train_dataset, val_dataset, test_dataset, train_indices, val_indices, test_indices = random_splits(
            dataset, 0.85, 0.10)
        with open('dataset_indices_random.pkl', 'wb') as f:
            pickle.dump({
                'train_indices': train_indices,
                'val_indices': val_indices,
                'test_indices': test_indices
            }, f)
    else:
        train_dataset, test_dataset, train_indices, test_indices = distance_splits(
            dataset)
        with open('dataset_indices.pkl', 'wb') as f:
            pickle.dump({
                'train_indices': train_indices,
                'test_indices': test_indices
            }, f)


if __name__ == '__main__':
    Fire(main)
