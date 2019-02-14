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

def five_fold_val(train_data, train_labels):
    length = len(train_data)
    indices = random.sample(range(length), int(length/5))
    data1 = []
    labels1 = []
    data5 = train_data
    labels5 = train_labels

    for i in sorted(indices, reverse = True):
        data1.append(train_data[1])
        labels1.append(train_labels[1])
        del data5[i]
        del labels5[i]

    indices = random.sample(range(len(data5)), int(length/5))
    data2 = []
    labels2 = []

    for i in sorted(indices, reverse = True):
        data2.append(data5[i])
        labels2.append(labels5[i])
        del data5[i]
        del labels5[i]

    indices = random.sample(range(len(data5)), int(length/5))
    data3 = []
    labels3 = []

    for i in sorted(indices, reverse = True):
        data3.append(data5[i])
        labels3.append(data5[i])
        del data5[i]
        del labels5[i]

    indices = random.sample(range(len(data5)), int(length / 5))
    data4 = []
    labels4 = []

    for i in sorted(indices, reverse = True):
        data4.append(data5[i])
        labels4.append(labels5[i])
        del data5[i]
        del labels5[i]

    return data1, labels1, data2, labels2, data3, labels3,
    data4, labels4, data5, labels5

def ten_fold_val(data, labels):
    datasets = []
    labelsets = []

    rem_data = data
    rem_labels = labels
    length = len(data)

    for i in range(9):
        indices = random.sample(range(len(rem_data)), int(length/10))
        tmpdata = []
        tmplabel = []
        for j in sorted(indices, reverse = True):
            tmpdata.append(rem_data[i])
            tmplabel.append(rem_labels[i])
            del rem_data[i]
            del rem_labels[i]
        datasets.append(tmpdata)
        labelsets.append(tmplabel)

    datasets.append(rem_data)
    labelsets.append(rem_labels)
    return datasets, labelsets


def LOO_val():
    return 0
