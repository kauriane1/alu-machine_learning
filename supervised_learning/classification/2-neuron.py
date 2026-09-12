def neuron_forward(X, weights, bias):
    """Calculates forward propagation predictions."""
    z = np.dot(X, weights) + bias
    predictions = 1 / (1 + np.exp(-z))  # Sigmoid activation function
    return predictions
