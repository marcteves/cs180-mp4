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

# create and standardize test and training vectors
for line in train_list.read().splitlines():
    train_vector.append(flat_greyscale_read(line))
train_vector = np.array(train_vector)
train_vector = StandardScaler().fit_transform(train_vector)

for line in test_list.read().splitlines():
    test_vector.append(flat_greyscale_read(line))
test_vector = np.array(test_vector)
test_vector = StandardScaler().fit_transform(test_vector)

features_size = train_vector.shape[1]

end = time.time()

print("Preprocessing done in %f" % (end - start))

# create and evaluate classifier for #1
mlp = MLPClassifier(hidden_layer_sizes = (int(features_size / 2)),
        solver='sgd')
start = time.time()
mlp.fit(train_vector, train_tags)
end = time.time()
print("Fit #1 done in %f" % (end - start))
accuracy = mlp.score(test_vector, test_tags)
print("Accuracy %f" % accuracy)

starting_size = features_size / 2

# store x and y values to graph later
x_values_mlp = []
y_values_mlp = []

# create and evaluate classifiers for #2
for num_neurons in np.linspace(starting_size, features_size, num=5):
    mlp = MLPClassifier(hidden_layer_sizes = (int(num_neurons)),
            solver='sgd')
    x_values_mlp.append(int(num_neurons))
    start = time.time()
    mlp.fit(train_vector, train_tags)
    end = time.time()
    print("Fit #2 done in %f" % (end - start))
    accuracy = mlp.score(test_vector, test_tags)
    y_values_mlp.append(accuracy)
    print("Accuracy %f" % accuracy)

# Reduce test and train vectors to ten dimensions (PCA #3)
# Standardize so that PCA doesn't get confused, then apply PCA
# keep top 10 principal components
pca = PCA(n_components = 10)
reduced_train_vector = pca.fit_transform(train_vector)
reduced_test_vector = pca.fit_transform(test_vector)
mlp = MLPClassifier(hidden_layer_sizes = (int(features_size / 2)),
        solver='sgd')
start = time.time()
mlp.fit(reduced_train_vector, train_tags)
end = time.time()
print("Fit #3 done in %f" % (end - start))
accuracy = mlp.score(reduced_test_vector, test_tags)
print("Accuracy %f" % accuracy)

# create classifier for #4
svm = SVC()
start = time.time()
svm.fit(train_vector, train_tags)
end = time.time()
print("Fit #4 done in %f" % (end - start))
accuracy = svm.score(test_vector, test_tags)
print("Accuracy %f" % accuracy)

x_values_svm = []
y_values_svm = []
# create classifiers for #5
for i in range(1,6):
    x_values_svm.append(i)
    start = time.time()
    svm = SVC(degree=i)
    end = time.time()
    print("Fit #5 done in %f" % (end - start))
    accuracy = svm.score(test_vector, test_tags)
    y_values_svm.append(accuracy)
    print("Accuracy %f" % accuracy)

x_values_svmg = []
y_values_svmg = []

for i in np.arange(0.1, 1.1, 0.1):
    x_values_svmg.append(i)
    start = time.time()
    svm = SVC(gamma=i)
    end = time.time()
    print("Fit done in %f" % (end - start))
    accuracy = svm.score(test_vector, test_tags)
    y_values_svmg.append(accuracy)
    print("Accuracy %f" % accuracy)

