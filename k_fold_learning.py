import os
import sys
import numpy as np
from data_splitting_functions import *
from id3_new import *

def k_fold(model, example, labels, fold=1):
    """

    :param model: A tree
    :param fold: n number of split size
    :return:
    """

    train, label_t, val, label_v = two_fold_val(example, labels)
    two_fold_id3 = ID3(train, label_t)
    two_fold_id3.build()

    correct_count = 0
    for i in range(len(val)):
        pred = two_fold_id3.classify_instance(val[i])
        # print("Prediction: ", pred, "  Actual: ", label2[i])
        if pred == label_v[i]:
            correct_count += 1

    print("Validation Accuracy 1: ", correct_count / len(val))

    return correct_count / len(val)
