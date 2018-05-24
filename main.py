#! /usr/bin/python3

import os
import sys
import argparse
import numpy as np
import time
import cv2
from sklearn.neural_network import MLPClassifier 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC

start = time.time()

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

end = time.time()

print("Preprocessing done in %f" % (end - start))

# create classifier for #1
mlp_1 = MLPClassifier(hidden_layer_sizes = (1, int(features_size / 2)),
        solver='sgd')
start = time.time()
mlp_1.fit(train_vector, train_tags)
end = time.time()
print("Fit #1 done in %f" % (end - start))

# create classifier for #2
mlp_2 = MLPClassifier(hidden_layer_sizes = (1, features_size),
        solver='sgd')
start = time.time()
mlp_2.fit(train_vector, train_tags)
end = time.time()
print("Fit #2 done in %f" % (end - start))

# Reduce test and train vectors to ten dimensions (PCA #3)
# Standardize so that PCA doesn't get confused, then apply PCA
# keep top 10 principal components
pca = PCA(n_components = 10)
reduced_train_vector = pca.fit_transform(
        StandardScaler().fit_transform(train_vector))
reduced_test_vector = pca.fit_transform(
        StandardScaler().fit_transform(test_vector))
mlp_3 = MLPClassifier(hidden_layer_sizes = (1, int(features_size / 2)),
        solver='sgd')
start = time.time()
mlp_3.fit(reduced_train_vector, train_tags)
end = time.time()
print("Fit #3 done in %f" % (end - start))

# create classifier for #4
svm_1 = SVC()
start = time.time()
svm_1.fit(train_vector, train_tags)
end = time.time()
print("Fit #4 done in %f" % (end - start))

# create classifiers for #5
svm_2 = []

for i in range(1,6):
    start = time.time()
    classifier = SVC(degree=i)
    svm_2.append(classifier)
    end = time.time()
    print("Fit #%d done in %f" % (4 + i, end - start))

# create classifiers for #6
svm_3 = []

for i in np.arange(0.1, 1.1, 0.1):
    start = time.time()
    classifier = SVC(gamma=i)
    svm_3.append(classifier)
    end = time.time()
    print("Fit done in %f" % (end - start))

# now use those classifiers we made
