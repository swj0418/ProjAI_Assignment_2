import os
import sys

from dataset import *
from id3lib import *

from k_fold_learning import *

if __name__ == '__main__':
    """
    ID3 Algorithm
    """

    k_fold_iterative_tennis(iterations=20)
