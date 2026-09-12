# --- COMPLETING THE MISSING VARIABLES ---
# N is the number of input features in the dataset
N = X_train.shape[1] 

# m is the number of samples
m = X_train.shape[0]

# Initialize weights with random values and biases with zeros
# w_ih is the matrix of weights from input to hidden layer
w_ih = np.random.randn(N, h_nodes) 
# b_h is the vector of hidden layer biases
b_h = np.zeros((1, h_nodes))
# w_ho is the matrix of weights from hidden layer to output
w_ho = np.random.randn(h_nodes, o_nodes)
# b_o is the vector of output layer biases
b_o = np.zeros((1, o_nodes))

# --- TRAINING LOOP ---
for epoch in range(epochs):
    # --- Forward Pass ---
    # z_h is the input to hidden layer
    z_h = np.dot(X_train, w_ih) + b_h
    # a_h is the activation of hidden layer
    a_h = sigmoid(z_h)
    
    # z_o is the input to output layer
    z_o = np.dot(a_h, w_ho) + b_o
    # a_o is the activation of output layer (predicted probability)
    a_o = sigmoid(z_o)
    
    # --- Backward Pass (Gradient Calculation) ---
    # error_o is the gradient at the output layer
    error_o = a_o - y_train
    # d_w_ho is the gradient for weights between hidden and output layer
    d_w_ho = np.dot(a_h.T, error_o) / m
    # d_b_o is the gradient for output biases
    d_b_o = np.sum(error_o, axis=0, keepdims=True) / m
    
    # error_h is the gradient propagated back to the hidden layer
    error_h = np.dot(error_o, w_ho.T) * sigmoid_derivative(a_h)
    # d_w_ih is the gradient for weights between input and hidden layer
    d_w_ih = np.dot(X_train.T, error_h) / m
    # d_b_h is the gradient for hidden biases
    d_b_h = np.sum(error_h, axis=0, keepdims=True) / m
    
    # --- Gradient Descent Weight Updates ---
    w_ho -= lr * d_w_ho
    b_o  -= lr * d_b_o
    w_ih -= lr * d_w_ih
    b_h  -= lr * d_b_h
