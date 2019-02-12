import os
import sys
import math

def get_entropy(set):
    """
    0 entropy means that all of the classifications in the data set are the same
    1 equal number of labels
    :param set:
    :return:
    """
    entropy = 0
    num_positive = 0
    num_negative = 0
    for element in set:
        if element == 0:
            num_negative += 1
        else:
            num_positive += 1

    entropy = -num_positive * math.log(num_positive, 2) - (num_negative * math.log(num_negative, 2))

    return entropy

def info_gain(set, A):
    """
    Information gain measures a reduction in entropy that results from partitioning the data on an attribute A.
    It represents how effective an attribute is at classifying the data.
    :param set: Set of attributes
    :param A: The attribute we are interested in
    :return:
    """
    set_size = len(set)
    entropy_S = get_entropy(set)

    attribute_count = 0
    for element in set:
        for attribute in element:
            if attribute == A: # Numbers
                attribute_count += 1