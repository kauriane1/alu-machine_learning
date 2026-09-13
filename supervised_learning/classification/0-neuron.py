#!/usr/bin/env python3
<<<<<<< HEAD

import numpy as np

Neuron = __import__('0-neuron').Neuron

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
print(neuron.W)
print(neuron.W.shape)
print(neuron.b)
print(neuron.A)
neuron.A = 10
print(neuron.A)
=======
"""Class Neuron that defines a single neuron performing binary classification
"""


import numpy as np


class Neuron:
    """ Class Neuron
    """

    def __init__(self, nx):
        """ Instantiation function of the neuron

        Args:
            nx (_type_): _description_

        Raises:
            TypeError: _description_
            ValueError: _description_
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        self.W = np.random.normal(size=(1, nx))
        self.b = 0
        self.A = 0
>>>>>>> 2c5a9ec7f8c108ca233a88ee94c694c1ecd10cb6
