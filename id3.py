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
        self.prediction = None # If prediction is not a NoneType, then it is a leaf
        self.children = [] # To keep track and traverse the tree
        self.next = None

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
        self.root = self.run_id3(self.data, self.label, target_attribute=0, attributes=[0, 1, 2, 3])
    """
    # ID3 Algorithm
    def run_id3(self, examples, labels, attribute):
        self.iteration += 1
        # Create root here?
        root = Node()

        # On the way, the tree generating algorithm will have to divide dataset
        if self.check_all_positive(labels):
            # Return root node with pos/yes label
            root.prediction = 1
            return root # , 1 # 1 as in positive label
        elif self.check_all_negative(labels):
            # Return root node with neg/no label
            root.prediction = 0
            return root # , 0 # Negative label
        elif len(attribute) == 0:
            # No attributes left
            # Return root node with label the most common value of classification attribute in example
            pred = self.get_most_popular(labels)
            root.prediction = pred
            return root # , self.get_most_popular(example_labels=labels)

        else:
            # Best attribute
            best_attribute, gain = get_highest_infogain(examples, labels, attribute)
            # Assign best attribute to the root node
            root.attribute_class = best_attribute

            # Caluculate different values
            num_distinct, values = self.extract_distinct_attributes(examples, best_attribute)
            for value in values:
                # For each attribute value for that specific best_attribute
                # e.g., Outlook [Sunny, Overcast, Rain] 0, 1, 2
                child_node = Node()
                child_node.assign_attribute_class(value)

                # Append children
                root.children.append(child_node)

                # Now determine and split examples those have value as their value for that attribute
                num_of_branch_examples, branch_example = \
                    self.who_has_this_value_as_its_attribute_class_value(data, best_attribute, value)

                new_attributes = attribute.pop(best_attribute)

                if num_of_branch_examples == 0:
                    # If branch is empty
                    child_node.prediction = self.get_most_popular(labels)
                else:
                    child_node = self.run_id3(branch_example, best_attribute, new_attributes)
        return root # Finally
    """
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
            root.prediction = 1
            return root  # , 1 # 1 as in positive label
        elif self.check_all_negative(labels):
            # Return root node with neg/no label
            root.prediction = 0
            return root  # , 0 # Negative label
        elif len(attributes) == 0:
            # No attributes left
            # Return root node with label the most common value of classification attribute in example
            pred = self.get_most_popular(labels)
            root.prediction = pred
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

                # Examples and labels now a subset
                num_of_branch_examples, branch_example, branch_label = \
                    self.who_has_this_value_as_its_attribute_class_value(examples, labels, best_attribute, value)

                # New attributes excluding the one used here
                new_attributes = []

                for i in range(len(attributes)):
                    if best_attribute != attributes[i]:
                        new_attributes.append(attributes[i])

                if num_of_branch_examples == 0:
                    # Empty
                    v = self.get_most_popular(labels)
                    node.assign_attribute_class(v)
                else:
                    # If evenly divided...
                    root.next = self.run_id3(branch_example, branch_label, target_attribute, new_attributes)

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
        self.root.print()
        for key in self.tree.keys():
            print(self.tree[key])

# root node contains
if __name__ == '__main__':
    data, label = load_tennis_dataset()
    id3 = ID3(data, label)
    id3.create_root()

    id3.print_tree()
