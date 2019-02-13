from dataset import *
from id3lib import *

# Node class : each node will be a class of attributes
#   (outlook, temperature, humidity, wind) having up to three children.
# param data : set of corresponding lists for days on which the
#    indicated attribute is present

class Node:
    def _init_(self, data):
        # self.cls = 0
        self.left = None
        self.middle = None
        self.right = None
        self.data = data

class ID3:
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def create_root(self):
        highest = get_highest_infogain(data, self.label, [0, 1, 2, 3])
        print(highest)

# root node contains
if __name__ == '__main__':
    data, label = load_tennis_dataset()
    id3 = ID3(data, label)
    id3.create_root()
