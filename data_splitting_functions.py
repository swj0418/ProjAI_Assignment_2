import os
import sys
import random

# Load in dataset being used
from dataset import *
data, label = load_tennis_dataset()

def split_train_test(data, label):
    length = len(data)
    ten_percent = int(length * 0.1)
    
