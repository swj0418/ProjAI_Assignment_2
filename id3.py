import PlayTennisDataset

# Node class : each node will be a class of attributes
#   (outlook, temperature, humidity, wind) having up to three children.
# param data : set of corresponding lists for days on which the
#    indicated attribute is present
class Node:

    def _init_(self, data):

        self.left = None
        self.middle = None
        self.right = None
        self.data = data

# root node contains
root = Node(PlayTennisDataset)

