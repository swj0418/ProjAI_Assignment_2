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
    def __init__(self, data, label, target):
        self.data = data
        self.label = label
        self.root = None
        self.target = target
        self.iteration = 0

    def build(self):
        self.root = self.run_id3(self.data, self.label, target_attribute=self.target, attributes=[0, 1, 2, 3])

    def predict(self, tree, attribute):
        print("===================== Prediction ======================")
        first_attribute_true = tree.attribute # Outlook
        value = attribute[first_attribute_true]
        print(first_attribute_true)
        for child in tree.children:
            edge_value = child.parent_attribute
            if edge_value is not None:
                # Directly to the answer. Directly to the leaf node
                if edge_value == value:
                    print("Prediction: ", child.value)
                    return child.value
            else:
                # If edge value is None
                # To the attribute node
                print("entered")
                self.predict(child, attribute)


    def run_id3(self, examples, labels, target_attribute, attributes, parent_attribute=None):
        """

        :param examples: Training examples
        :param target_attribute: Attribute whose value is to be predicted by the tree
        :param attributes: A list of other attributes that may be tested by the learned decision tree
        :return:
        """
        root = Node()
        # print(examples, "    ", labels, "    ", target_attribute, "    ", attributes)

        if self.check_all_positive(labels):
            # Return root node with pos/yes label
            root.parent_attribute = parent_attribute
            root.value = 1
            # print("All positive: ", root.value)
            return root  # , 1 # 1 as in positive label
        elif self.check_all_negative(labels):
            # Return root node with neg/no label
            root.parent_attribute = parent_attribute
            root.value = 0
            # print("All negative: ", root.value)
            return root  # , 0 # Negative label
        elif len(attributes) == 0:
            # No attributes left
            # Return root node with label the most common value of classification attribute in example
            pred = self.get_most_popular(labels)
            # print("Most common attribute: ", pred)
            root.parent_attribute = parent_attribute
            root.value = pred
            return root  # , self.get_most_popular(example_labels=labels)

        else:
            best_attribute, gain = get_highest_infogain(examples, labels, attributes)
            root.attribute = best_attribute

            num_distinct, possible_values = self.extract_distinct_attributes(examples, best_attribute)
            for value in possible_values:
                # Add a new tree branch below root. Children list
                node = Node()
                node.attribute = value
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
                    # print("Calling iter: ", self.iteration, attributes, "    ", new_attributes, "   ", best_attribute, "   ", retrieve_tennis_attribute_label(best_attribute))
                    # Below this new branch add a subtree
                    self.iteration += 1
                    node = self.run_id3(branch_example, branch_label, target_attribute, new_attributes, parent_attribute=value)
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
        print("Depth 0: ", "parent attribute: ", self.root.parent_attribute, "  my_attribute: ", self.root.attribute,
              "  my_value: ", self.root.value)
        for child in self.root.children:
            print("Depth 1: ", "parent attribute: ", child.parent_attribute, "  my_attribute: ", child.attribute, "  my_value: ", child.value)

            for c in child.children:
                print("Depth 2: ", "parent attribute: ", c.parent_attribute, "  my_attribute: ", c.attribute, "  my_value: ", c.value)
                for final in c.children:
                    print("Depth 3: ", "parent attribute: ", final.parent_attribute, "  my_attribute: ", final.attribute,
                          "  my_value: ", final.value)


# root node contains
if __name__ == '__main__':
    data, label = load_tennis_dataset()
    # data, label = load_id3_test_dataset()

    id3 = ID3(data, label, target=[0, 0, 1, 0])
    id3.build()

    id3.print_tree()
    id3.predict(id3.root, [0, 1, 1, 1])
