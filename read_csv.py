import os
import numpy as np


def read_csv(p=2, training=True):
    """
    Reads test or training CSV and returns it as two-dimensional numpy arrays
    :param training: Boolean telling wetter or not to read training data
    :return: Xi, Yi parameters
    """
    path = os.path.join(os.path.dirname(__file__), 'data', 'data-train.csv')
    if not training:
        path = os.path.join(os.path.dirname(__file__), 'data', 'data-test.csv')

    x_parameters = []
    y_parameters = []
    with open(path) as file:
        for line in file:
            all_parameters = line.strip().split(',')
            x_parameters.append(all_parameters[:p])
            y_parameters.append(all_parameters[p:])
    return np.array(x_parameters, dtype=float), np.array(y_parameters, dtype=float)
    #return np.genfromtxt('data/data-train.csv', delimiter=',')

