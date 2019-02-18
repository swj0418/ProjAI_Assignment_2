import os
import sys
import numpy as np
from data_splitting_functions import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from id3_new import *

def k_fold(model, example, labels, k=2):
    """

    :param model: A tree
    :param fold: n number of split size
    :return: accuracy and the tree
    """
    train_accuracies = []
    validation_accuracies = []

    if k == 2:
        train, label_t, val, label_v = two_fold_val(data, labels)
    elif k == 5:
        train, label_t, val, label_v = five_fold_val(data, labels)
    elif k == 10:
        train, label_t, val, label_v = ten_fold_val(data, labels)
    else:
        train, label_t, val, label_v = LOO_val(data, labels)

    folded_id3 = ID3(train, label_t)
    for i in range(k):
        folded_id3.incremental_build(i)

        correct_count = 0
        for i in range(len(train)):
            pred = folded_id3.classify_instance(train[i])
            if pred == label_t[i]:
                correct_count += 1

        train_accuracy = correct_count / len(train)

        correct_count = 0
        for i in range(len(val)):
            pred = folded_id3.classify_instance(val[i])
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


def k_fold_iterative_custom(iterations=100, k = 2):
    data, label, test_example, test_labels = load_custom_dataset()
    train_accuracies = []
    validation_accuracies = []

    if k == 2:
        train, label_t, val, label_v = two_fold_val(data, labels)
    elif k == 5:
        train, label_t, val, label_v = five_fold_val(data, labels)
    elif k == 10:
        train, label_t, val, label_v = ten_fold_val(data, labels)
    else:
        train, label_t, val, label_v = LOO_val(data, labels)

    folded_id3 = ID3(train, label_t)
    for i in range(iterations):
        folded_id3.incremental_build(i)

        correct_count = 0
        for i in range(len(train)):
            pred = folded_id3.classify_instance(train[i])
            if pred == label_t[i]:
                correct_count += 1

        train_accuracy = correct_count / len(train)

        correct_count = 0
        for i in range(len(val)):
            pred = folded_id3.classify_instance(val[i])
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


# Create trees and validation testing using two_fold validation
def two_fold_test(data, labels):
    # get two_fold datasets
    data1, label1, data2, label2 = two_fold_val(data, labels)

    # create the trees
    tree1 = ID3(data1, label1)
    tree1.build()
    tree2 = ID3(data2, label2)
    tree2.build()

    print("Starting two-fold cross-validation")
    print("----------------------------------")
    # Accuracy Checks for Tree 1
    correct_count = 0
    for i in range(len(data2)):
        pred = tree1.classify_instance(data2[i])
        # print("Prediction: ", pred, "  Actual: ", label_v[i])
        if pred == label2[i]:
            correct_count += 1

    validation_accuracy = correct_count / len(data2)

    print("Accuracy 1: ", validation_accuracy)
    print()

    # Accuracy Checks for Tree 2
    correct_count = 0
    for i in range(len(data1)):
        pred = tree2.classify_instance(data1[i])
        # print("Prediction: ", pred, "  Actual: ", label_v[i])
        if pred == label1[i]:
            correct_count += 1

    validation_accuracy = correct_count / len(data1)

    print("Accuracy 2: ", validation_accuracy)
    print()
    print('---------------------------------')
    print()
    print()


def five_fold_test(data, labels):
    datasets, labelsets = five_fold_val(data, labels)
    print("Starting Five-Fold Cross-Validation")
    print('-----------------------------------')
    accuracies = []

    count = 1
    for i in range(len(datasets)):
        val_data = datasets[i]
        val_labels = labelsets[i]

        tmp_data = []
        tmp_labels = []
        for j in range(len(datasets)):
            if i != j:
                for x in range(len(datasets[j])):
                    tmp_data.append(datasets[j][x])
                    tmp_labels.append(labelsets[j][x])

        tree = ID3(tmp_data, tmp_labels)
        tree.build()

        correct_count = 0
        for i in range(len(val_data)):
            pred = tree.classify_instance(val_data[i])
            # print("Prediction: ", pred, "  Actual: ", label_v[i])
            if pred == val_labels[i]:
                correct_count += 1

        validation_accuracy = correct_count / len(val_data)
        accuracies.append(validation_accuracy)


    print('---------------------------------------------')

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.plot([1,2,3,4,5],accuracies, 'r')

    plt.show()

def ten_fold_test(data, labels):
    datasets, labelsets = ten_fold_val(data, labels)
    print("Starting Ten-Fold Cross-Validation")
    print('-----------------------------------')
    accuracies = []

    count = 1
    for i in range(len(datasets)):
        val_data = datasets[i]
        val_labels = labelsets[i]

        tmp_data = []
        tmp_labels = []
        for j in range(len(datasets)):
            if i != j:
                for x in range(len(datasets[j])):
                    tmp_data.append(datasets[j][x])
                    tmp_labels.append(labelsets[j][x])

        tree = ID3(tmp_data, tmp_labels)
        tree.build()

        correct_count = 0
        for i in range(len(val_data)):
            pred = tree.classify_instance(val_data[i])
            # print("Prediction: ", pred, "  Actual: ", label_v[i])
            if pred == val_labels[i]:
                correct_count += 1

        validation_accuracy = correct_count / len(val_data)
        accuracies.append(validation_accuracy)


    print('---------------------------------------------')

    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.plot( [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], accuracies, 'r')

    plt.show()


def LOO_test(data, labels):
    trainsets, label_t_sets, validation, label_v = LOO_val(data, labels)
    accuracies = []

    for i in range(len(trainsets)):
        tree = ID3(trainsets[i], label_t_sets[i])
        tree.build()

        pred = tree.classify_instance(validation[i])
        if pred == label_v[i]:
            accuracies.append(1)
        else:
            accuracies.append(0)

    accuracy = sum(accuracies) / float(len(accuracies))
    print("LOO Accuracy: ", accuracy)
