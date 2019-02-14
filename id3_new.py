from dataset import *
from id3lib import *
from data_splitting_functions import *

import numpy as np
# Node class : each node will be a class of attributes
#   (outlook, temperature, humidity, wind) having up to three children.
# param data : set of corresponding lists for days on which the
#    indicated attribute is present


class Node:
    # Node will be universal whether they are root or not.
    # No need to mark root
    def __init__(self):
        self.attribute = None
        self.parent_attribute = None
        self.value = None
        self.children = []
        self.sibling = []

    def print(self):
        if self.attribute == None:
            print("Value: ", self.value)
        elif self.value == None:
            print("My attribute: ", self.attribute)
            for child in self.children:
                child.print()

class ID3:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.root = None
        self.iteration = 0

    def build(self):
        self.root = self.run_id3(self.data, self.label, range(len(data[0])))

    def predict(self, attribute):
        print("===================== Prediction ======================")
        print("Depth 0: ", "parent attribute: ", self.root.parent_attribute, "  my_attribute: ", self.root.attribute,
              "  my_value: ", self.root.value)

        for child in self.root.children:
            print("Depth 1: ", "parent attribute: ", child.parent_attribute, "  my_attribute: ", child.attribute,
                  "  my_value: ", child.value)

            for c in child.children:
                print("Depth 2: ", "parent attribute: ", c.parent_attribute, "  my_attribute: ", c.attribute,
                      "  my_value: ", c.value)
                for final in c.children:
                    print("Depth 3: ", "parent attribute: ", final.parent_attribute, "  my_attribute: ",
                          final.attribute,
                          "  my_value: ", final.value)

    def run_id3(self, examples, labels, attributes, parent_attribute=None):
        """

        :param examples: Training examples
        :param attributes: A list of other attributes that may be tested by the learned decision tree
        :return:
        """
        root = Node()
        root.parent_attribute = parent_attribute
        print(examples, "    ", labels, "    ", attributes)

        if self.check_all_positive(labels):
            # Return root node with pos/yes label
            root.value = 1
            print("All positive: ", root.value)
            return root  # , 1 # 1 as in positive label
        elif self.check_all_negative(labels):
            # Return root node with neg/no label
            root.value = 0
            print("All negative: ", root.value)
            return root  # , 0 # Negative label
        elif len(attributes) == 0:
            # No attributes left
            # Return root node with label the most common value of classification attribute in example
            pred = self.get_most_popular(labels)
            print("Most common attribute: ", pred)
            root.value = pred
            return root  # , self.get_most_popular(example_labels=labels)

        else:
            best_attribute, gain = get_highest_infogain(examples, labels, attributes)
            root.attribute = best_attribute

            num_distinct, possible_values = self.extract_distinct_attributes(examples, best_attribute)
            for value in possible_values:
                # Add a new tree branch below root. Children list
                node = Node()
                node.attribute = best_attribute
                node.value = value

                # print("New node value: ", value)

                # Examples and labels now a subset
                num_of_branch_examples, branch_example, branch_label = \
                    self.who_has_this_value_as_its_attribute_class_value(examples, labels, best_attribute, value)

                # New attributes excluding the one used here
                new_attributes = []

                for i in range(len(attributes)):
                    if best_attribute != attributes[i]:
                        new_attributes.append(attributes[i])
                # print("New attributes: ", branch_example)

                if num_of_branch_examples == 0:
                    # Empty
                    v = self.get_most_popular(labels)
                    node.value = v
                else:
                    print("Calling iter: ", self.iteration, attributes, "    ", new_attributes, "   ", best_attribute, "   ", retrieve_tennis_attribute_label(best_attribute))
                    # Below this new branch add a subtree
                    self.iteration += 1
                    node = self.run_id3(branch_example, branch_label, new_attributes, parent_attribute=value)
                    root.children.append(node)

        return root

    def check_all_positive(self, example_labels):
        tmp = np.asarray(example_labels)
        avg = tmp.mean()
        if avg == 1:
            return True
        else:
            return False

    def check_all_negative(self, example_labels):
        tmp = np.asarray(example_labels)
        avg = tmp.mean()
        if avg == 0:
            return True
        else:
            return False

    def get_most_popular(self, example_labels):
        tmp = np.asarray(example_labels)
        avg = tmp.mean()
        if avg >= 0.5:
            return 1 # positive labels superiority
        else:
            return 0

    def extract_distinct_attributes(self, examples, label):
        countered = []
        for instance in examples:
            if instance[label] not in countered:
                countered.append(instance[label])

        number_of_distinct = len(countered)
        return number_of_distinct, countered

    def who_has_this_value_as_its_attribute_class_value(self, examples, labels, best_attribute, value):
        branch_example = []
        branch_labels = []
        count = 0
        for instance in examples:
            if instance[best_attribute] == value:
                branch_example.append(instance)
                branch_labels.append(labels[count])
            count += 1

        num_of_examples = len(branch_example)
        return num_of_examples, branch_example, branch_labels

    def print_tree(self):
        print("===================== Decision TREE ======================")
        # Attribute class
        print(self.root.attribute)
        self.root.print()
        print("==========================================================")
        print(self.root.attribute)
        print("Depth 0: ", "parent attribute value: ", self.root.parent_attribute, "  my_attribute: ", self.root.attribute,
              "  my_value: ", self.root.value)
        for child in self.root.children:
            print("Depth 1: ", "parent attribute value: ", child.parent_attribute, "  my_attribute: ", child.attribute, "  my_value: ", child.value)

            for c in child.children:
                print("Depth 2: ", "parent attribute value: ", c.parent_attribute, "  my_attribute: ", c.attribute, "  my_value: ", c.value)
                for final in c.children:
                    print("Depth 3: ", "parent attribute value: ", final.parent_attribute, "  my_attribute: ", final.attribute,
                          "  my_value: ", final.value)


    # Method for classifying new instances
    def classify_instance(self, example):
        cur_node = self.root

        # traverse the tree until we reach a leaf node
        while cur_node.value is None:

            # get value for the attribute in the desired instance
            # for the current decision node
            cur_attribute = cur_node.attribute
            attribute_val = example[cur_attribute]

            # find the child node down the path with the correct value
            for i in cur_node.children:
                if i.parent_attribute == attribute_val:
                    cur_node = i
                    break

        # value at leaf node is our classification
        return cur_node.value






# root node contains
if __name__ == '__main__':

    # read in data, label sets
    data, label = load_tennis_dataset()

    # split data into training and testing sets
    train_data, train_labels, test_data, test_labels = split_train_test(data, label)

    # two-fold validation testing
    print("Starting Two-Fold Validation Testing")
    print()
    data1, label1, data2, label2 = two_fold_val(train_data, train_labels)
    first_two_fold_id3 = ID3(data1, label1)
    first_two_fold_id3.build()
    print("======== First Iteration =========")
    for i in range(len(data2)):
        pred = first_two_fold_id3.classify_instance(data2[i])
        print("Prediction: ", pred, "  Actual: ", label2[i])
    print()
    print("======== Second Iteration ======= ")
    second_two_fold_id3 = ID3(data2, label2)
    second_two_fold_id3.build()

    for i in range(len(data1)):
        pred = second_two_fold_id3.classify_instance(data1[i])
        print("Prediction: ", pred, "  Actual: ", label1[i])


    '''
    id3 = ID3(data, label)
    id3.build()

    id3.print_tree()

    for i in range(len(data)):
        pred = id3.classify_instance(data[i])
        print("Prediction: ", pred, "  Actual: ",label[i])
        '''
