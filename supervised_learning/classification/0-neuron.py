import numpy as np

class Neuron:
    """Defines a single neuron performing binary classification."""
    
    def __init__(self, nx):
        """Initializes the neuron."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        
        # Initialize weights using a standard normal distribution
        self.W = np.random.randn(1, nx)
        # Initialize bias to 0
        self.b = 0
        # Initialize activated output to 0
        self.A = 0
