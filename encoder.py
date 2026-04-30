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