#! /usr/bin/python3

import os
import sys
import argparse
import numpy as np
import cv2
from sklearn.neural_network import MLPClassifier 

def flat_greyscale_read(filename):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE).flatten()

parser = argparse.ArgumentParser(description="""Takes four files as input,
        returns a vector classifying the test set""")
parser.add_argument("train_set")
parser.add_argument("train_tags")

parser.add_argument("test_set")
parser.add_argument("test_tags")

args = parser.parse_args()

train_list = open(args.train_set)
test_list = open(args.test_set)
train_tags_file = open(args.train_tags)
test_tags_file = open(args.test_tags)

train_vector = []
train_tags = []
test_vector = []
test_tags = []

# create tags vectors
for line in train_tags_file.read().splitlines():
    train_tags.append(int(line))
train_tags = np.array(train_tags)

for line in test_tags_file.read().splitlines():
    test_tags.append(int(line))
test_tags = np.array(test_tags)

# create test and training vectors
for line in train_list.read().splitlines():
    train_vector.append(flat_greyscale_read(line))
train_vector = np.array(train_vector)

for line in test_list.read().splitlines():
    test_vector.append(flat_greyscale_read(line))
test_vector = np.array(test_vector)

features_size = train_vector.shape[1]

mlp_1 = MLPClassifier(hidden_layer_sizes = (1, int(features_size / 2)),
        solver='lbfgs')
mlp_1.fit(train_vector, train_tags)


mlp_2 = MLPClassifier(hidden_layer_sizes = (1, features_size),
        solver='lbfgs')
mlp_2.fit(train_vector, train_tags)
