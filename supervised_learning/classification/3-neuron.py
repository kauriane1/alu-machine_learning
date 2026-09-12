def neuron_cost(predictions, y):
    """Calculates the cost (binary cross-entropy) for predictions."""
    m = y.shape[0]
    cost = -1/m * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost
