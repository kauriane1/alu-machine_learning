import numpy as np

class Neuron:
    """Defines a single neuron performing binary classification with private attributes."""
    
    def __init__(self, nx):
        """Initializes the private neuron."""
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be positive")
        
        # Initialize private attributes
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """Getter for the private weights vector."""
        return self.__W

    @property
    def b(self):
        """Getter for the private bias."""
        return self.__b

    @property
    def A(self):
        """Getter for the private activated output."""
        return self.__A
