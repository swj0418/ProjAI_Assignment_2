import os
import sys

def load_tennis_dataset():
    set = [[2, 2, 2, 0],
		     [2, 2, 2, 2],
		     [0, 2, 2, 0],
		     [1, 1, 2, 0],
		     [1, 0, 0, 0],
		     [1, 0, 0, 2],
		     [0, 0, 0, 2],
		     [2, 1, 2, 0],
		     [2, 0, 0, 0],
		     [1, 1, 0, 0],
		     [2, 1, 0, 2],
		     [0, 1, 2, 2],
		     [0, 2, 0, 0],
		     [1, 1, 2, 2]]

    labels = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]

    return set, labels


def load_custom_dataset():
    set = []
    labels = []

    return set, labels