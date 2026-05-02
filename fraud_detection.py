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

# Builds the encoder
def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Function initializes encoder weights
def initialize_encoder(input_dim, hidden_dims):
    weights = []
    biases = []

    prev_dim = input_dim

    for dim in hidden_dims:
        weight = np.random.randn(prev_dim, dim) * np.sqrt(2.0 / prev_dim)
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

# DECODER 

def initialize_decoder(latent_dim, hidden_dims, output_dim):
    weights = [] 
    biases = []
    prev_dim = latent_dim   # starting from bottleneck size(8)
    
    for layer_size in hidden_dims: 
        weight = np.random.randn(prev_dim, layer_size) * np.sqrt(2.0 / prev_dim)
        bias = np.zeros((1,layer_size))
        weights.append(weight)
        biases.append(bias)
        prev_dim = layer_size 
    # The output layer (linear, no ReLU values can be negative after scaling them)
    weight = np.random.randn(prev_dim, output_dim) * np.sqrt(2.0 / prev_dim)
    bias = np.zeros((1, output_dim))
    weights.append(weight)
    biases.append(bias)
    return weights, biases

# Forward pass
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

# MSE LOSS
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

# test cases
input_dim = x_train.shape[1] #30
latent_dim = 8
hidden_dims = [16] # mirror of encoder hidden layers in reverse

dec_weights, dec_biases = initialize_decoder(latent_dim, hidden_dims, input_dim)
reconstructed, dec_activations = decoder_forward(
    latent_representation,
    dec_weights,
    dec_biases
)

loss = mse_loss(x_train, reconstructed)
print("Reconstructed shape: ", reconstructed.shape)
print("Initial MSE loss:", loss)

# ENCODER BACKWARD
def encoder_backward(grad_input, activations, weights):
    grad_weights = []
    grad_biases = []
    grad = grad_input
    for i in reversed(range(len(weights))):
        A_prev = activations[i]
        n = A_prev.shape[0]
        dW = np.dot(A_prev.T, grad) / n
        db = np.mean(grad, axis=0, keepdims=True)
        grad_weights.insert(0, dW)
        grad_biases.insert(0, db)
        if i > 0:
            grad = np.dot(grad, weights[i].T)
            grad = grad * relu_derivative(activations[i])
    return grad_weights, grad_biases

# TRAINING LOOP
def train_autoencoder(x_train, encoder_weights, encoder_biases, dec_weights, dec_biases, epochs=50, learning_rate=0.001, batch_size=256, patience=10):
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    n_samples = x_train.shape[0]
    for epoch in range(epochs):
        # Shuffles data each epoch
        indices = np.random.permutation(n_samples)
        x_shuffled = x_train[indices]
        epoch_loss = 0
        n_batches = 0
        for start in range(0, n_samples, batch_size):
            x_batch = x_shuffled[start:start + batch_size]
            # Forward pass
            latent, enc_activations = encoder_forward(x_batch, encoder_weights, encoder_biases)
            reconstructed, dec_activations = decoder_forward(latent, dec_weights, dec_biases)

            # Loss
            loss = mse_loss(x_batch, reconstructed)
            epoch_loss += loss
            n_batches += 1

            # Backward pass
            grad_output = mse_loss_gradient(x_batch, reconstructed)
            grad_output = np.clip(grad_output, -1.0, 1.0)
            dec_grad_weights, dec_grad_biases, grad_input = decoder_backward(grad_output, dec_activations, dec_weights)
            grad_input = np.clip(grad_input, -1.0, 1.0)
            enc_grad_weights, enc_grad_biases = encoder_backward(grad_input, enc_activations, encoder_weights)

            # Updates weights
            for i in range(len(dec_weights)):
                dec_weights[i] -= learning_rate * dec_grad_weights[i]
                dec_biases[i] -= learning_rate * dec_grad_biases[i]
            for i in range(len(encoder_weights)):
                encoder_weights[i] -= learning_rate * enc_grad_weights[i]
                encoder_biases[i] -= learning_rate * enc_grad_biases[i]

        avg_loss = epoch_loss / n_batches
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
    return encoder_weights, encoder_biases, dec_weights, dec_biases, losses

# Runs training
print("\nTraining autoencoder...")
encoder_weights, encoder_biases, dec_weights, dec_biases, losses = train_autoencoder(
    x_train, encoder_weights, encoder_biases, dec_weights, dec_biases, epochs=300, learning_rate=0.003, patience=20
)

# ANOMALY DETECTION
print("\nRunning anomaly detection...")
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Gets reconstruction error for the test set (normal transactions)
latent_test, _ = encoder_forward(x_test, encoder_weights, encoder_biases)
reconstructed_test, _ = decoder_forward(latent_test, dec_weights, dec_biases)
test_errors = np.mean((x_test - reconstructed_test) ** 2, axis=1)

# Tests ALL transactions (normal + fraud)
x_all = scaler.transform(x.values)
latent_all, _ = encoder_forward(x_all, encoder_weights, encoder_biases)
reconstructed_all, _ = decoder_forward(latent_all, dec_weights, dec_biases)
all_errors = np.mean((x_all - reconstructed_all) ** 2, axis=1)

# Threshold tuning analysis
print("\nThreshold Tuning Analysis:")
print(f"{'Percentile':<15} {'Threshold':<15} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print("-" * 65)

for percentile in [90, 93, 95, 97, 99]:
    thresh = np.percentile(test_errors, percentile)
    y_pred_thresh = (all_errors > thresh).astype(int)
    p = precision_score(y, y_pred_thresh, zero_division=0)
    r = recall_score(y, y_pred_thresh, zero_division=0)
    f = f1_score(y, y_pred_thresh, zero_division=0)
    print(f"{percentile:<15} {thresh:<15.6f} {p:<12.4f} {r:<12.4f} {f:<12.4f}")

# Using 95th percentile as final threshold
threshold = np.percentile(test_errors, 95)
print(f"\nFinal Anomaly Threshold (95th percentile): {threshold:.6f}")

# Flags anything above threshold as fraud
y_pred = (all_errors > threshold).astype(int)

# EVALUATION
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

print("\nEvaluation Results:")
print(f"Precision: {precision_score(y, y_pred):.4f}")
print(f"Recall: {recall_score(y, y_pred):.4f}")
print(f"F1 Score: {f1_score(y, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y, all_errors):.4f}")

cm = confusion_matrix(y, y_pred)
print(f"\nConfusion Matrix:")
print(f"True Negatives  (Normal correctly identified): {cm[0][0]}")
print(f"False Positives (Normal flagged as fraud):     {cm[0][1]}")
print(f"False Negatives (Fraud missed):                {cm[1][0]}")
print(f"True Positives  (Fraud correctly caught):      {cm[1][1]}")

# PLOTS LOSS CURVE
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig("loss_curve.png")
print("\nLoss curve saved as loss_curve.png")