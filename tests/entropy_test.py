import os
import sys

from id3lib import *
from dataset import *

def entropy_test(X, Y):
    print("Entropy Test")

    total_set_entropy = get_entropy(Y) # Put label
    print("Total set entropy: ", total_set_entropy)

def info_gain_test(X, Y, attribute):
    print("Info Gain Test")
    gain = info_gain(X, Y, attribute)
    print(gain)

if __name__ == '__main__':
    """
    Entropy Test
    """
    X, Y = load_tennis_dataset()
    entropy_test(X, Y)
    info_gain_test(X, Y, 0)
