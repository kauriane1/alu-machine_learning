def evaluate_neuron(X, y, weights, bias):
    """Performs forward pass and computes the resulting cost."""
    preds = neuron_forward(X, weights, bias)
    cost = neuron_cost(preds, y)
    return preds, cost
