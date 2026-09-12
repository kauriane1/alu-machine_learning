#!/usr/bin/env python3

import numpy as np


class DeepNeuralNetwork:
    """Defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Initialize the deep neural network"""

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        for nodes in layers:
            if not isinstance(nodes, int) or nodes < 1:
                raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i, nodes in enumerate(layers):
            if i == 0:
                previous_nodes = nx
            else:
                previous_nodes = layers[i - 1]

            self.weights["W{}".format(i + 1)] = (
                np.random.randn(nodes, previous_nodes)
                * np.sqrt(2 / previous_nodes)
            )

            self.weights["b{}".format(i + 1)] = np.zeros((nodes, 1))
