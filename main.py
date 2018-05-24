#! /usr/bin/python3

import os
import argparse
import numpy as np
import time
import cv2
import plotly.offline as plt
import plotly.graph_objs as go
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
    svm.fit(train_vector, train_tags)
    end = time.time()
    print("Fit #5 done in %f" % (end - start))
    accuracy = svm.score(test_vector, test_tags)
    y_values_svm.append(accuracy)
    print("Accuracy %f" % accuracy)

x_values_svmg = []
y_values_svmg = []
# create classifiers for #5

for i in np.arange(0.1, 1.1, 0.1):
    x_values_svmg.append(i)
    start = time.time()
    svm = SVC(gamma=i)
    svm.fit(train_vector, train_tags)
    end = time.time()
    print("Fit done in %f" % (end - start))
    accuracy = svm.score(test_vector, test_tags)
    y_values_svmg.append(accuracy)
    print("Accuracy %f" % accuracy)


# make graphs
# graph for #2

x_values_mlp = np.array(x_values_mlp)
y_values_mlp = np.array(y_values_mlp)

trace_2 = go.Scatter(
        x = x_values_mlp,
        y = y_values_mlp,
        mode = 'lines+markers',
        line = dict(
            shape = "spline"
            )
        )

layout_2 = go.Layout(
        title = 'Accuracy graph',
        xaxis = dict(
            title = '# of nodes in hidden layer',
            ),
        yaxis = dict(
            title = 'accuracy',
            ),
        )

figure_2 = go.Figure(data = [trace_2], layout = layout_2)
plt.plot(figure_2, filename='figure_2.html')

# graph for #5

x_values_svm = np.array(x_values_svm)
y_values_svm = np.array(y_values_svm)

trace_5 = go.Scatter(
        x = x_values_svm,
        y = y_values_svm,
        mode = 'lines+markers',
        line = dict(
            shape = "spline"
            )
        )

layout_5 = go.Layout(
        title = 'Accuracy graph',
        xaxis = dict(
            title = 'degree of polynomial',
            ),
        yaxis = dict(
            title = 'accuracy',
            ),
        )

figure_5 = go.Figure(data = [trace_5], layout = layout_5)
plt.plot(figure_5, filename='figure_5.html')

# graph for #6

x_values_svmg = np.array(x_values_svmg)
y_values_svmg = np.array(y_values_svmg)

trace_6 = go.Scatter(
        x = x_values_svmg,
        y = y_values_svmg,
        mode = 'lines+markers',
        line = dict(
            shape = "spline"
            )
        )

layout_6 = go.Layout(
        title = 'Accuracy graph',
        xaxis = dict(
            title = 'gamma value',
            ),
        yaxis = dict(
            title = 'accuracy',
            ),
        )

figure_6 = go.Figure(data = [trace_6], layout = layout_6)
plt.plot(figure_6, filename='figure_6.html')
