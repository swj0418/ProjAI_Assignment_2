import os
import sys
import numpy as np
from data_splitting_functions import *
from id3_new import ID3
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from id3_new import *

def k_fold(model, example, labels, fold=1):
    """

    :param model: A tree
    :param fold: n number of split size
    :return: accuracy and the tree
    """

    train, label_t, val, label_v = two_fold_val(example, labels)
    two_fold_id3 = ID3(train, label_t)
    two_fold_id3.build()

    correct_count = 0
    for i in range(len(train)):
        pred = two_fold_id3.classify_instance(train[i])
        if pred == label_t[i]:
            correct_count += 1

    train_accuracy = correct_count / len(train)

    correct_count = 0
    for i in range(len(val)):
        pred = two_fold_id3.classify_instance(val[i])
        # print("Prediction: ", pred, "  Actual: ", label_v[i])
        if pred == label_v[i]:
            correct_count += 1

    validation_accuracy = correct_count / len(val)

    print("Training Accuracy: ", train_accuracy)
    print("Validation Accuracy: ", validation_accuracy)

    return validation_accuracy, two_fold_id3.root


def k_fold_iterative_custom(iterations=100):
    data, label, test_example, test_labels = load_custom_dataset()
    train_accuracies = []
    validation_accuracies = []

    train, label_t, val, label_v = two_fold_val(data, labels)
    two_fold_id3 = ID3(train, label_t)
    for i in range(iterations):
        two_fold_id3.incremental_build(i)

        correct_count = 0
        for i in range(len(train)):
            pred = two_fold_id3.classify_instance(train[i])
            if pred == label_t[i]:
                correct_count += 1

        train_accuracy = correct_count / len(train)

        correct_count = 0
        for i in range(len(val)):
            pred = two_fold_id3.classify_instance(val[i])
            # print("Prediction: ", pred, "  Actual: ", label_v[i])
            if pred == label_v[i]:
                correct_count += 1

        validation_accuracy = correct_count / len(val)

        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        print("Training Accuracy: ", train_accuracy)
        print("Validation Accuracy: ", validation_accuracy)

    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.plot(train_accuracies, 'g')
    plt.plot(validation_accuracies, 'r')

    red_patch = mpatches.Patch(color='red', label='Validation Accuracy')
    green_patch = mpatches.Patch(color='green', label='Train Accuracy')
    plt.legend(handles=[red_patch, green_patch])

    plt.show()

def k_fold_iterative_tennis(iterations=100):
    data, label = load_tennis_dataset()
    train_accuracies = []
    validation_accuracies = []

    train, label_t, val, label_v = two_fold_val(data, labels)
    two_fold_id3 = ID3(train, label_t)
    for i in range(iterations):
        two_fold_id3.incremental_build(i)

        correct_count = 0
        for i in range(len(train)):
            pred = two_fold_id3.classify_instance(train[i])
            if pred == label_t[i]:
                correct_count += 1

        train_accuracy = correct_count / len(train)

        correct_count = 0
        for i in range(len(val)):
            pred = two_fold_id3.classify_instance(val[i])
            # print("Prediction: ", pred, "  Actual: ", label_v[i])
            if pred == label_v[i]:
                correct_count += 1

        validation_accuracy = correct_count / len(val)

        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        print("Training Accuracy: ", train_accuracy)
        print("Validation Accuracy: ", validation_accuracy)

    plt.xlabel("iterations")
    plt.ylabel("accuracy")
    plt.plot(train_accuracies, 'g')
    plt.plot(validation_accuracies, 'r')

    red_patch = mpatches.Patch(color='red', label='Validation Accuracy')
    green_patch = mpatches.Patch(color='green', label='Train Accuracy')
    plt.legend(handles=[red_patch, green_patch])

    plt.show()

    return two_fold_id3, correct_count / len(val)

def breast_cancer_incremental_build():
    data, label, test_example, test_labels = load_custom_dataset()
    train_accuracies = []
    validation_accuracies = []
    train_error_rates = []
    validation_error_rates = []

    train, label_t, validation, label_v = two_fold_val(data, labels)
    tree = ID3(train, label_t)
    for i in range(len(train)):
        # tree.incremental_build_v2(train[i], label_t[i])
        tree.incremental_build(i)

        correct_count = 0
        error = 0
        for i in range(len(train)):
            pred = tree.classify_instance(train[i])
            if pred == label_t[i]:
                correct_count += 1
            else:
                error += 1

        train_accuracy = correct_count / len(train)
        train_error_rate = error / len(train)

        correct_count = 0
        error = 0
        for i in range(0, len(validation)):
            pred = tree.classify_instance(validation[i])
            # print("Prediction: ", pred, "  Actual: ", label_v[i])
            if pred == label_v[i]:
                correct_count += 1
            else:
                error += 1

        validation_accuracy = correct_count / len(validation)
        validation_error_rate = error / len(validation)

        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)
        train_error_rates.append(train_error_rate)
        validation_error_rates.append(validation_error_rate)

        # print("Training Accuracy: ", train_accuracy)
        # print("Validation Accuracy: ", validation_accuracy)
        print("iteration: ", i)

    plt.xlabel("iterations")
    plt.ylabel("error")
    plt.plot(train_error_rates, 'g')
    plt.plot(validation_error_rates, 'r')

    red_patch = mpatches.Patch(color='red', label='Validation Error Rate')
    green_patch = mpatches.Patch(color='green', label='Train Error Rate')
    plt.legend(handles=[red_patch, green_patch])

    plt.show()

    tree.root.print(0)
