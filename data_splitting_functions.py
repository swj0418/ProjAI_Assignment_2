import os
import sys
import random

# Load in dataset being used
from dataset import *
data, labels = load_custom_dataset()

def split_train_test(data, labels):
    # find 10% of total size of dataset
    length = len(data)
    ten_percent = int(length * 0.1)

    # randomly select indices of 10% of set
    indices = random.sample(range(length), ten_percent)

    test_data = []
    test_labels = []
    train_data = data
    train_labels = labels

    for i in sorted(indices, reverse = True):
        test_data.append(data[i])
        test_labels.append(labels[i])
        del train_data[i]
        del train_labels[i]

    return train_data, train_labels, test_data, test_labels

split_train_test(data, labels)
