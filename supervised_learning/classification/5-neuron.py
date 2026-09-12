import numpy as np

class Neuron:
    def __init__(self, nx):
        """
        Initializes the neuron.
        nx is the number of input features.
        """
        # Initializing weights randomly and bias to 0 as standard practice
        self.__W = np.random.randn(1, nx)
        self.__b = 0.0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron.
        
        Parameters:
        X (numpy.ndarray): shape (nx, m) containing the input data
        Y (numpy.ndarray): shape (1, m) containing the correct labels
        A (numpy.ndarray): shape (1, m) containing the activated output
        alpha (float): learning rate
        """
        # m is the number of examples
        m = X.shape[1]
        
        # Calculate gradients
        dZ = A - Y
        dW = (1 / m) * np.dot(dZ, X.T)
        db = (1 / m) * np.sum(dZ)
        
        # Update private attributes
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
