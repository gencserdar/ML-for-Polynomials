import numpy as np
import pandas as pd
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(project_dir, "polynomials.csv")

# Load the dataset
dataset = pd.read_csv(file_path)

# Extract and normalize coefficients from the dataset
real_data = dataset[[col for col in dataset.columns if col.startswith('a')]].values
real_data = np.nan_to_num(real_data)
real_data = np.clip(real_data, -1e10, 1e10)
real_data = (real_data - np.mean(real_data)) / (np.std(real_data) + 1e-8)  # Z-score normalization

# Hyperparameters
latent_dim = 16
output_dim = real_data.shape[1]
hidden_dim = 64
learning_rate_gen = 0.00005
learning_rate_disc = 0.0001
epochs = 1000
batch_size = 64
gradient_clip_value = 1.0
x_range = (-5, 5)

# Activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def ensure_positive(coeffs, x_range):
    x = np.linspace(*x_range, 100)
    poly_values = sum(c * (x ** i) for i, c in enumerate(coeffs))
    if np.any(poly_values < 0):
        coeffs = np.abs(coeffs) + 1e-3
    return coeffs

def clip_gradients(grad):
    return np.clip(grad, -gradient_clip_value, gradient_clip_value)

# Generator model
class Generator:
    def __init__(self, latent_dim, output_dim, hidden_dim):
        self.W1 = np.random.randn(latent_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    def forward(self, z):
        self.z1 = np.dot(z, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.z2  # Linear output
        return self.a2

    def backward(self, z, grad_output):
        dz2 = grad_output * relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dz1 = np.dot(dz2, self.W2.T) * relu_derivative(self.z1)
        dW1 = np.dot(z.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W1 -= learning_rate_gen * clip_gradients(dW1)
        self.b1 -= learning_rate_gen * clip_gradients(db1)
        self.W2 -= learning_rate_gen * clip_gradients(dW2)
        self.b2 -= learning_rate_gen * clip_gradients(db2)

# Discriminator model
class Discriminator:
    def __init__(self, input_dim, hidden_dim):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return np.clip(self.a2, 1e-7, 1 - 1e-7)

    def backward(self, x, grad_output):
        dz2 = grad_output * sigmoid_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dz1 = np.dot(dz2, self.W2.T) * relu_derivative(self.z1)
        dW1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W1 -= learning_rate_disc * clip_gradients(dW1)
        self.b1 -= learning_rate_disc * clip_gradients(db1)
        self.W2 -= learning_rate_disc * clip_gradients(dW2)
        self.b2 -= learning_rate_disc * clip_gradients(db2)

# Loss Functions
def positivity_penalty(coeffs, x_range=(-10, 10), num_samples=100):
    x = np.linspace(*x_range, num_samples)
    poly_values = np.polyval(coeffs[::-1], x)
    penalty = np.sum(np.maximum(-poly_values, 0))
    return penalty

def generator_loss(discriminator, fake_data, lambda_penalty=10.0):
    g_pred = discriminator.forward(fake_data)
    adversarial_loss = -np.mean(np.log(g_pred + 1e-8))
    penalty = np.mean([positivity_penalty(fd) for fd in fake_data])
    total_loss = adversarial_loss + lambda_penalty * penalty
    return total_loss

def train_model(generator, discriminator):
    for epoch in range(epochs):
        z = np.random.randn(batch_size, latent_dim)
        fake_data = generator.forward(z)

        real_data_batch = real_data[np.random.choice(real_data.shape[0], batch_size, replace=True)]
        combined_data = np.vstack((real_data_batch, fake_data))
        labels = np.vstack((np.ones((batch_size, 1)), np.zeros((batch_size, 1))))

        d_pred = discriminator.forward(combined_data)
        d_loss = -np.mean(labels * np.log(d_pred + 1e-8) + (1 - labels) * np.log(1 - d_pred + 1e-8))
        discriminator.backward(combined_data, d_pred - labels)

        accuracy = np.mean((d_pred > 0.5) == labels)

        g_loss = generator_loss(discriminator, fake_data)
        grad_output = np.dot(discriminator.forward(fake_data) - 1, discriminator.W2.T)[:, :output_dim]
        generator.backward(z, grad_output)

        if epoch % 50 == 0:
            print(f"Epoch {epoch}/{epochs} - D Loss: {d_loss:.4f}, G Loss: {g_loss:.4f}, D Accuracy: {accuracy:.4f}")

# Generate polynomial
def generate_poly_with_gan(degree):
    generator = Generator(latent_dim, output_dim, hidden_dim)
    discriminator = Discriminator(output_dim, hidden_dim)
    
    train_model(generator, discriminator)

    polynomial = generator.forward(np.random.randn(1, latent_dim)).flatten()
    if len(polynomial) < degree + 1:
        additional_coeffs = np.random.uniform(-1, 1, degree + 1 - len(polynomial))
        polynomial = np.concatenate((polynomial, additional_coeffs))
    else:
        polynomial = polynomial[:degree + 1]

    polynomial = ensure_positive(polynomial, x_range)
    print("Generated coefficients:", polynomial * 100)
    return polynomial * 100
