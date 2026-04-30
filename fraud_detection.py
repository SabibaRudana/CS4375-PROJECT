import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


url = "https://detect-credit-card-fraud.s3.us-east-2.amazonaws.com/creditcard.csv"
df = pd.read_csv(url)

# Preprocessing
x = df.drop("Class", axis=1)
y = df["Class"]
x_normal = x[y == 0]    # train only on non-fraud cases

x_train, x_test = train_test_split(x_normal, test_size=0.2, random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Build the encoder
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Initialize encoder weights
def initialize_encoder(input_dim, hidden_dims):
    weights = []
    biases = []

    prev_dim = input_dim

    for dim in hidden_dims:
        weight = np.random.randn(prev_dim, dim) * 0.01
        bias = np.zeros((1, dim))

        weights.append(weight)
        biases.append(bias)
        prev_dim = dim

    return weights, biases

# Forward Pass
def encoder_forward(x, weights, biases):
    activations = [x]

    A = x   # A = current activation (starts as input)
    for W, b in zip(weights, biases):
        Z = np.dot(A, W) + b    # linear combination
        A = relu(Z)
        activations.append(A)

    return A, activations   # A = compressed representation

if __name__ == "__main__":

    # testing
    input_dim = x_train.shape[1]
    hidden_dims = [16, 8]

    encoder_weights, encoder_biases = initialize_encoder(input_dim, hidden_dims)

    latent_representation, activations = encoder_forward(
        x_train,
        encoder_weights,
        encoder_biases
    )

    print("Latent shape:", latent_representation.shape) # (n_samples, 8)



#DECODER 

def initialize_decoder(latent_dim, hidden_dims, output_dim):
    weights = [] 
    biases = []
    prev_dim = latent_dim   #starting from bottleneck size(8)
    
    for layer_size in hidden_dims: 
        weight = np.random.randn(prev_dim, layer_size) * 0.01
        bias = np.zeros((1,layer_size))
        weights.append(weight)
        biases.append(bias)
        prev_dim = layer_size 
    # The output layer (linear, no ReLU values can be negative after scaling them)
    weight = np.random.randn(prev_dim, output_dim) * 0.01
    bias = np.zeros((1, output_dim))
    weights.append(weight)
    biases.append(bias)
    return weights, biases

#Forward pass
def decoder_forward(z, weights, biases) :
    activations = [z]
    A = z # A = current activation
    for W, b in zip(weights[:-1], biases[:-1]):
        Z = np.dot(A, W) + b #linear combination 
        A = relu(Z)
        activations.append(A)

    Z_out = np.dot(A, weights[-1]) + biases[-1]
    activations.append(Z_out)
    return Z_out, activations

#MSE LOSS
def mse_loss(x_original, x_reconstructed):
    return np.mean((x_original - x_reconstructed) ** 2)

def mse_loss_gradient(x_original, x_reconstructed):
    n = x_original.shape[0]
    return(2/n) * (x_reconstructed - x_original)

#Backward pass 
def decoder_backward(grad_output, activations, weights):
    grad_weights = [] 
    grad_biases = []
    grad = grad_output 
    for i in reversed(range(len(weights))):
        A_prev = activations[i]
        n = A_prev.shape[0]
        dW = np.dot(A_prev.T, grad) / n
        db = np.mean(grad, axis = 0, keepdims = True)
        grad_weights.insert(0, dW)
        grad_biases.insert(0, db)
        if i > 0:
            grad = np.dot(grad, weights[i].T)
            grad = grad * relu_derivative(activations[i])
    grad_input = np.dot(grad, weights[0].T)
    return grad_weights, grad_biases, grad_input

#test cases
input_dim = x_train.shape[1] #30
latent_dim = 8
hidden_dims = [16] #mirror of encoder hidden layers in reverse

dec_weights, dec_biases = initialize_decoder(latent_dim, hidden_dims, input_dim)
reconstructed, dec_activations = decoder_forward(
    latent_representation,
    dec_weights,
    dec_biases
)

loss = mse_loss(x_train, reconstructed)
print("Reconstructed shape: ", reconstructed.shape)
print("Initial MSE loss:", loss)