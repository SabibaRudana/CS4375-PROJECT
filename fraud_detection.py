import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

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

def log_experiment(file_name, params, results):
    # Count existing experiments
    try:
        with open(file_name, "r") as f:
            lines = f.readlines()
            exp_count = sum(1 for line in lines if line.startswith("Experiment"))
    except FileNotFoundError:
        exp_count = 0

    exp_number = exp_count + 1

    with open(file_name, "a") as f:
        f.write(f"\n{'='*40}\n")
        f.write(f"Experiment {exp_number}\n")
        f.write("Parameters Chosen:\n")

        for key, value in params.items():
            f.write(f"{key}: {value}\n")

        f.write("\nResults:\n")

        for key, value in results.items():
            f.write(f"{key}: {value}\n")

if __name__ == "__main__":

    np.random.seed(42)

    # Initialize
    input_dim = x_train.shape[1]
    latent_dim = 8
    encoder_hidden = [16, 8]
    decoder_hidden = [16]

    encoder_weights, encoder_biases = initialize_encoder(input_dim, encoder_hidden)
    dec_weights, dec_biases = initialize_decoder(latent_dim, decoder_hidden, input_dim)

    # Train
    print("\nTraining autoencoder...")
    encoder_weights, encoder_biases, dec_weights, dec_biases, losses = train_autoencoder(
        x_train,
        encoder_weights,
        encoder_biases,
        dec_weights,
        dec_biases,
        epochs=300,
        learning_rate=0.003,
        patience=20
    )

    # Anomaly Detection
    print("\nRunning anomaly detection...")

    latent_test, _ = encoder_forward(x_test, encoder_weights, encoder_biases)
    reconstructed_test, _ = decoder_forward(latent_test, dec_weights, dec_biases)
    test_errors = np.mean((x_test - reconstructed_test) ** 2, axis=1)

    x_all = scaler.transform(x)
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

    threshold = np.percentile(test_errors, 97)
    y_pred = (all_errors > threshold).astype(int)

    # Evaluation
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc = roc_auc_score(y, all_errors)

    print("\nEvaluation Results:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

    cm = confusion_matrix(y, y_pred)

    print("\nConfusion Matrix:")
    print(f"True Negatives  (Normal correctly identified): {cm[0][0]}")
    print(f"False Positives (Normal flagged as fraud):     {cm[0][1]}")
    print(f"False Negatives (Fraud missed):                {cm[1][0]}")
    print(f"True Positives  (Fraud correctly caught):      {cm[1][1]}")

    # Logging
    params = {
        "Number of Layers": "3 (Encoder + Decoder)",
        "Neurons": "(30, 16, 8, 16, 30)",
        "Error Function": "MSE",
        "Learning Rate": 0.003,
        "Epochs": 300,
        "Batch Size": 256,
        "Train/Test Split": "80:20",
        "Dataset Size": x.shape[0],
        "Threshold": f"{threshold:.6f}"
    }

    results = {
        "Precision": f"{precision:.4f}",
        "Recall": f"{recall:.4f}",
        "F1 Score": f"{f1:.4f}",
        "ROC-AUC": f"{roc:.4f}",
        "True Negatives": cm[0][0],
        "False Positives": cm[0][1],
        "False Negatives": cm[1][0],
        "True Positives": cm[1][1]
    }

    log_experiment("experiment_log.txt", params, results)

    print("\nExperiment log saved.")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.title("Training Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.savefig("loss_curve.png")
    print("\nLoss curve saved as loss_curve.png")