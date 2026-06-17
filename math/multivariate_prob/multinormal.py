#!/usr/bin/env python3
"""Multivariate Normal Distribution"""


import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """
        Class constructor

        data is a numpy.ndarray of shape (d, n)
        """

        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)

        X = data - self.mean

        self.cov = np.matmul(X, X.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point
        """

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.mean.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)

        inv = np.linalg.inv(self.cov)

        diff = x - self.mean

        exponent = -0.5 * np.matmul(
            np.matmul(diff.T, inv),
            diff
        )

        denominator = np.sqrt(
            ((2 * np.pi) ** d) * det
        )

        pdf = np.exp(exponent) / denominator

        return float(pdf)
