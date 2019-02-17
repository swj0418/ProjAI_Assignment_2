import os
import sys
import math
import cmath

def get_entropy(labels):
    """
    0 entropy means that all of the classifications in the data set are the same
    1 equal number of labels
    :param set:
    :return:
    """
    total = len(labels)
    num_positive = 0
    num_negative = 0
    for element in labels:
        if element == 0:
            num_negative += 1
        else:
            num_positive += 1
    try:
        entropy = -(num_positive / total) * math.log(num_positive / total, 2) - ((num_negative / total) * math.log(num_negative / total, 2))
    except:
        entropy = 0

    return entropy

def info_gain(set, label, A):
    """
    Information gain measures a reduction in entropy that results from partitioning the data on an attribute A.
    It represents how effective an attribute is at classifying the data.
    :param set: Set of attributes
    :param A: The attribute we are interested in [0, 1, 2, 3] [Outlook, temp, humid, wind]
    :return:
    """
    set_size = len(set)
    entropy_S = get_entropy(label) # Entropy for the whole set

    # Determine how many values there are for this specific attribute
    diff = []
    for element in set:
        for attribute in element:
            if element[A] not in diff:
                diff.append(element[A]) # element[A] refers to specific value for that attribute

    # Number of different values e.g., Outlook = 3, Weather = 2
    num_values = max(diff)

    """
    This part requires generalization
    This works only for tennis
    """
    attribute_counts = []
    set_attributes_pair = []
    set_labels_pair = []

    for i in range(num_values + 1):
        attribute_counts.append(0)
        set_attributes_pair.append([])
        set_labels_pair.append([])

    element_count = 0
    for element in set:
        count = 0
        for attribute in element:
            if count == A:
                attribute_counts[attribute] += 1
                set_attributes_pair[attribute].append(element)
                set_labels_pair[attribute].append(label[element_count])

            count += 1
        element_count += 1

    return_value = entropy_S
    # Calculate individual entropies
    count = 0
    for label_set in set_labels_pair:
        entropy = get_entropy(label_set)
        return_value -= (attribute_counts[count] / set_size) * entropy
        count += 1

    # print("attribute counts: ", attribute_counts)
    # print("Entropy: ", entropy_S)
    return return_value

def get_highest_infogain(set, label, attribute_classes):
    max = 0
    attribute_class_num = 0
    count = 0
    for attribute_class in attribute_classes:
        gain = info_gain(set, label, attribute_class)
        if gain > max:
            max = gain
            attribute_class_num = attribute_classes[count]

        count += 1
    # print("MAX: ", max, "    class num: ", attribute_class_num)

    return attribute_class_num, gain