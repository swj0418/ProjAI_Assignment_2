import os
import sys

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

    # Open file with breast cancer data
    fileref = open(os.path.join(sys.path[0], "breast-cancer-wisconsin.data"), "r")

    # Iterate through each line of file
    for line in fileref:
        line = line.rstrip() # remove EOL chars
        line_vals = line.split(",") # seperate the features

        # Convert all features in the current line to integers
        # if an NA value is found (represented as '?' in dataset), exclude from
        # reature vector
        for i in range(0, len(line_vals)):
            if line_vals[i] == '?':
                break
            line_vals[i] = int(line_vals[i])
            # check if we're at last element
            # won't make it here is there is NA value
            if i == len(line_vals) - 1:
                set.append(line_vals)

    # Label extraction
    for i in range(0, len(set)):
        x = set[i][10]
        if x == 2:
            x = 0
        else:
            x = 1
        labels.append(x)

    # Remove labels from feature vectors
    tmp = []

    # iterate through all elements in feature array
    # remove last value (label)
    for element in set:
        tmp.append(element[0:10])
    set = tmp

    return set, labels
