#!/usr/bin/env python3

import numpy as np


def one_hot_decode(one_hot):
    """Converts a one-hot matrix into a vector of labels"""

    if not isinstance(one_hot, np.ndarray):
        return None

    if one_hot.ndim != 2:
        return None

    return np.argmax(one_hot, axis=0)
