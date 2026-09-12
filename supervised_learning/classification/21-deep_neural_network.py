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

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i, nodes in enumerate(layers):
            if i == 0:
                previous_nodes = nx
            else:
                previous_nodes = layers[i - 1]

            self.__weights["W{}".format(i + 1)] = (
                np.random.randn(nodes, previous_nodes)
                * np.sqrt(2 / previous_nodes)
            )

            self.__weights["b{}".format(i + 1)] = np.zeros((nodes, 1))

    @property
    def L(self):
        """Getter for L"""
        return self.__L

    @property
    def cache(self):
        """Getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """Getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation"""

        self.__cache["A0"] = X

        for i in range(1, self.__L + 1):
            W = self.__weights["W{}".format(i)]
            b = self.__weights["b{}".format(i)]
            A_prev = self.__cache["A{}".format(i - 1)]

            Z = np.matmul(W, A_prev) + b
            A = 1 / (1 + np.exp(-Z))

            self.__cache["A{}".format(i)] = A

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the neural network"""

        m = Y.shape[1]

        return -(1 / m) * np.sum(
            Y * np.log(A) +
            (1 - Y) * np.log(1.0000001 - A)
        )

    def evaluate(self, X, Y):
        """Evaluates the neural network"""

        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)

        prediction = (A >= 0.5).astype(int)

        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent"""

        m = Y.shape[1]
        dZ = cache["A{}".format(self.__L)] - Y

        for layer in range(self.__L, 0, -1):
            A_prev = cache["A{}".format(layer - 1)]
            W = self.__weights["W{}".format(layer)]

            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if layer > 1:
                dZ = np.matmul(W.T, dZ) * A_prev * (1 - A_prev)

            self.__weights["W{}".format(layer)] -= alpha * dW
            self.__weights["b{}".format(layer)] -= alpha * db
