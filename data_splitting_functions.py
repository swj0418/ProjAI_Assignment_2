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

def two_fold_val(train_data, train_labels):
    # find size of training set
    length = len(train_data)

    # randomly seperate in half
    indices = random.sample(range(length), int(length / 2))

    data1 = []
    labels1 = []
    data2 = train_data
    labels2 = train_labels

    for i in sorted(indices, reverse = True):
        data1.append(train_data[i])
        labels1.append(train_labels[i])
        del data2[i]
        del labels2[i]
        
    return data1, labels1, data2, labels2

two_fold_val(data, labels)

def five_fold_val():
    return 0

def ten_fold_val():
    return 0

def LOO_val():
    return 0
