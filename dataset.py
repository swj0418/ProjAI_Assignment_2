import os
import sys
import csv
import numpy as np

def load_id3_test_dataset():
    set = [[2, 1, 0, 1],
           [0, 1, 1, 0],
           [1, 0, 1, 2],
           [1, 1, 0, 0],
           [2, 1, 2, 1]]

    labels = [0, 1, 0, 1, 0]

    return set, labels


def load_tennis_dataset():
    set = [[2, 2, 1, 0],
		     [2, 2, 1, 1],
		     [0, 2, 1, 0],
		     [1, 1, 1, 0],
		     [1, 0, 0, 0],
		     [1, 0, 0, 1],
		     [0, 0, 0, 1],
		     [2, 1, 1, 0],
		     [2, 0, 0, 0],
		     [1, 1, 0, 0],
		     [2, 1, 0, 1],
		     [0, 1, 1, 1],
		     [0, 2, 0, 0],
		     [1, 1, 1, 1]]

    labels = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]

    return set, labels


def retrieve_tennis_attribute_label(attribute_class_id):
    tennis_attribute_name = {0: "Outlook", 1: "Temperature", 2: "Humidity", 3: "Wind"}
    if attribute_class_id > 3 or attribute_class_id < 0:
        return "Bad Attribute Id"
    else:
        return tennis_attribute_name[attribute_class_id]


def load_custom_dataset():
    set = []
    labels = []

    """
    Training dataset
    """
    # Open file with breast cancer data
    fileref = open(os.path.join(sys.path[0], "breast-cancer-wisconsin.data"), "r")

    # Iterate through each line of file
    for line in fileref:
        line = line.rstrip() # remove EOL chars
        line_vals = line.split(",") # seperate the features
        set.append(line_vals[1:]) # Exclude sample id
        labels.append(line_vals[len(line_vals) - 1:])

    examples = []
    adjusted_labels = []
    for i in range(len(set)):
        if "?" not in set[i]:
            examples.append(list(map(int, set[i])))
            if int(labels[i][0]) == 2:
                adjusted_labels.append(0)
            else:
                adjusted_labels.append(1)

    """
    Test dataset
    """
    test_example = []
    test_label = []
    with open("./breast_cancer_wisconsin-test.data", "r") as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if "?" in row:
                break
            else:
                test_example.append(list(map(int, row[1:len(row) - 1])))
                label = row[len(row) - 1:]
                test_label.append(int(label[0]))

    return examples, adjusted_labels, test_example, test_label
