from dataset import *
from id3lib import *

# Node class : each node will be a class of attributes
#   (outlook, temperature, humidity, wind) having up to three children.
# param data : set of corresponding lists for days on which the
#    indicated attribute is present


class Node:
    def __init__(self, data, attribute_class):
        self.attribute_class = attribute_class # ID of the attribute class
        self.attribute_name = retrieve_tennis_attribute_label(attribute_class) # Name of the attribute class
        self.left = None
        self.middle = None
        self.right = None
        self.data = data


class ID3:
    def __init__(self, data, label):
        self.data = data
        self.label = label
        # Tree. Dict.
        # parent : [node]
        # parent will be marked with integers 0, 1, 2, 3
        # 0: Overlook, 1: Temperature, 2: Humidity, 3: Wind
        # For tennis example
        # The topmost root's parent is -1
        self.tree = {}

    def create_root(self):
        significant_attribute, gain = get_highest_infogain(data, self.label, [0, 1, 2, 3])
        # This will be the root
        self.tree[-1] = Node(self.data, significant_attribute)

    def generate_tree(self):
        # On the way, the tree generating algorithm will have to divide dataset
        pass

    def print_tree(self):
        for key in self.tree.keys():
            print(self.tree[key].attribute_name)


# root node contains
if __name__ == '__main__':
    data, label = load_tennis_dataset()
    id3 = ID3(data, label)
    id3.create_root()

    id3.print_tree()
