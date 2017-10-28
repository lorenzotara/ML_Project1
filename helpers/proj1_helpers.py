# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from matplotlib import pyplot as plt
from helpers.proj1_functions import *


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def predict_labels_for_lr(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred < 0.5)] = -1
    y_pred[np.where(y_pred >= 0.5)] = 1

    return y_pred


def plot_gamma_parameter(x_train, y_train, x_test, y_test, max_iters, initial_w, gamma_start=0, gamma_end=10, step_size=1):

    axis_x = []
    axis_y = []

    for gamma in range(gamma_start, gamma_end, step_size):

        losses, ws = linear_regression(y_train, x_train, initial_w, max_iters, gamma / 100)

        y_pred = predict_labels(ws, x_test)

        final_result = y_test == y_pred

        score = np.count_nonzero(final_result) / len(final_result)

        axis_x.append(gamma / 100)
        axis_y.append(score)

        print(score * 100, "%")

    print("\n\n\n")
    print(axis_y)
    print(axis_x)

    plt.figure()
    plt.plot(axis_x, axis_y)
    plt.xlabel("gamma")
    plt.ylabel("score")
    plt.show()


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
