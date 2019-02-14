import os
import sys
import random

# Load in dataset being used
from dataset import *
data, labels = load_custom_dataset()

# Create and return a training and testing set from the dataset provided
# The test set will be selected by randomly selecting 10% of the instances
# from the total dataset
# The return is four arrays: one containing the feature vectors for all
# instances in the training set, one containing the corresponding labels
# for all instances in the training set, and then the same two arrays containing
# the information for the testing set.
def split_train_test(data, labels):
    # find 10% of total size of dataset
    length = len(data)
    ten_percent = int(length * 0.1)

    # randomly select indices of 10% of set
    indices = random.sample(range(length), ten_percent)

    test_data = []
    test_labels = []
    train_data = data.copy()
    train_labels = labels.copy()

    # for each random index add that data to the testing set and remove
    # from training. Indices are sorted and removed in reverse order so
    # that the indexing will not be affected by changes in size due to
    # removing items.
    for i in sorted(indices, reverse = True):
        test_data.append(data[i])
        test_labels.append(labels[i])
        del train_data[i]
        del train_labels[i]

    return train_data, train_labels, test_data, test_labels

# Separates training set for two-fold validation through
# random selection (and removal) of half of the data
# The return is four arrays: two contain the group of feature vectors for
# the two different groups, and the other two containing the corresponding
# labels
def two_fold_val(train_data, train_labels):
    # find size of training set
    length = len(train_data)

    # randomly seperate in half
    indices = random.sample(range(length), int(length / 2))

    data1 = []
    labels1 = []
    data2 = train_data.copy()
    labels2 = train_labels.copy()

    for i in sorted(indices, reverse = True):
        data1.append(train_data[i])
        labels1.append(train_labels[i])
        del data2[i]
        del labels2[i]

    return data1, labels1, data2, labels2

# Creates dataset for 5-fold validation by randomly breaking the
# training set into 5 (roughly) equal-sized chunks
# The return is two arrays: one containing each of the five sets of
# data groups, and the other containg the corresponding labels
def five_fold_val(data, labels):
    datasets = []
    labelsets = []

    rem_data = data.copy()
    rem_labels = labels.copy()
    length = len(data)

    for i in range(4):
        indices = random.sample(range(len(rem_data)), int(length/5))
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

# Creates datasets for 10-fold validation by randomly breaking the
# training set into 10 (roughly) equal-sized sections
# The return is two arrays: one containing each of the ten sets of
# data groups, and the other containg the corresponding labels
def ten_fold_val(data, labels):
    datasets = []
    labelsets = []

    rem_data = data.copy()
    rem_labels = labels.copy()
    length = len(data)

    # for the first 9 datasets, keep randomly selecting a number of
    # indices equal to 1/10 the size of the original dataset
    # removing these from the remaining data will result in the remaining
    # data after all iterations being the correct size for the 10th section
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


# Creates datasets for leave-one-out cross validation
# Goes through all possible iterations of leaving one instance out of the
# training set
# Returns 4 arrays: two containing the feature vectors of the validation and
# training data, and two containing the labels. The arrays are organized so that
# training_data[0] will have the feature vectors for all instances except
# the instance at validation_data[0], and so on.
def LOO_val(data, labels):
    validation_data = []
    validation_labels = []
    training_data = []
    training_labels = []

    for i in range(len(data)):
        tmp_data = data.copy()
        tmp_labels = labels.copy()

        validation_data.append(tmp_data[i])
        validation_labels.append(tmp_labels[i])

        del tmp_data[i]
        del tmp_labels[i]

        training_data.append(tmp_data)
        training_labels.append(tmp_labels)

    return training_data, training_labels, validation_data, validation_labels
