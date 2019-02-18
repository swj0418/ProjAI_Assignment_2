import os
import sys

from dataset import *
from id3lib import *
from id3_new import ID3
from k_fold_learning import *

if __name__ == '__main__':
    """
    ID3 Algorithm
    """

    """
    Play tennis basic run
    # train_ex, train_l, test_ex, test_l = load_custom_dataset()
    train_ex, train_l = load_tennis_dataset()

    tree = ID3(train_ex, train_l)

    tree.build()
    tree.root.print(0)
    
    pred = tree.classify_instance([2, 2, 1, 1])
    print(pred)
    """

    breast_cancer_incremental_build()
