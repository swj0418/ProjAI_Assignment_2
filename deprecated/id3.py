from dataset import *
from id3lib import *

import numpy as np
# Node class : each node will be a class of attributes
#   (outlook, temperature, humidity, wind) having up to three children.
# param data : set of corresponding lists for days on which the
#    indicated attribute is present


class Node:
    # Node will be universal whether they are root or not.
    # No need to mark root
    def __init__(self):
        self.attribute_class = None # ID of the attribute class
        self.attribute_name = None # Name of the attribute class
        # self.prediction = None # If prediction is not a NoneType, then it is a leaf
        self.children = [] # To keep track and traverse the tree
        self.sibling = None

    def assign_attribute_class(self, attribute_id):
        self.attribute_class = attribute_id
        self.attribute_name = retrieve_tennis_attribute_label(attribute_id)

    def print(self):
        print(self.prediction)
        if len(self.children) != 0:
            for child in self.children:
                child.print()

        else:
            print("  ====================================  ")
            print("Attribute Name: ", self.attribute_name, "  Label: ", self.attribute_class)


class ID3:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.root = None
        self.attributes = []
        # Tree. Dict.
        # parent : [node]
        # parent will be marked with integers 0, 1, 2, 3
        # 0: Overlook, 1: Temperature, 2: Humidity, 3: Wind
        # For tennis example
        # The topmost root's parent is -1
        self.tree = {}
        self.iteration = 0

    def create_root(self):
        best_attribute, gain = get_highest_infogain(data, self.label, [0, 1, 2, 3])
        # self.tree[-1] = Node(best_attribute)
        self.root = self.run_id3(self.data, self.label, target_attribute=[0, 0, 0, 0], attributes=[0, 1, 2, 3])

    def run_id3(self, examples, labels, target_attribute, attributes):
        """

        :param examples: Training examples
        :param target_attribute: Attribute whose value is to be predicted by the tree
        :param attributes: A list of other attributes that may be tested by the learned decision tree
        :return:
        """
        root = Node()
        #print(examples, "    ", labels, "    ", target_attribute, "    ", attributes)

        if self.check_all_positive(labels):
            # Return root node with pos/yes label
            root.attribute_class = 1
            print("All positive: ", root.attribute_class)
            return root  # , 1 # 1 as in positive label
        elif self.check_all_negative(labels):
            # Return root node with neg/no label
            root.attribute_class = 0
            print("All negative: ", root.attribute_class)
            return root  # , 0 # Negative label
        elif len(attributes) == 0:
            # No attributes left
            # Return root node with label the most common value of classification attribute in example
            pred = self.get_most_popular(labels)
            print("Most common attribute: ", pred)
            root.attribute_class = pred
            return root  # , self.get_most_popular(example_labels=labels)

        else:
            best_attribute, gain = get_highest_infogain(examples, labels, attributes)
            root.assign_attribute_class(best_attribute)

            num_distinct, possible_values = self.extract_distinct_attributes(examples, best_attribute)
            for value in possible_values:
                # Add a new tree branch below root. Children list
                node = Node()
                node.assign_attribute_class(value)
                root.children.append(node)
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
                    node.sibling = v
                else:
                    print("Calling", attributes, "    ", new_attributes, "   ", best_attribute, "   ", retrieve_tennis_attribute_label(best_attribute))
                    # Below this new branch add a subtree
                    root.sibling = self.run_id3(branch_example, branch_label, target_attribute, new_attributes)

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

        '''
        print(retrieve_tennis_attribute_label(self.root.attribute_class))
        for child in self.root.children:
            print(str(child.attribute_class) + "  |", end='\t')

        print()
        for child in self.root.sibling.children:
            print(child.attribute_class, end="\t")

        print()

        for child in self.root.sibling.children:
            print(child)

        print("\n")
        '''
        print(self.root.attribute_class)
        print(self.root.children)

# root node contains
if __name__ == '__main__':
    # data, label = load_tennis_dataset()
    data, label = load_id3_test_dataset()

    id3 = ID3(data, label)
    id3.create_root()

    id3.print_tree()
