import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)


# Calculate activation function values
random_values =  np.linspace(-5, 5, 100)

sigmoid_y = sigmoid(random_values)
relu_y = relu(random_values)
leaky_relu_y = leaky_relu(random_values)
tanh_y = tanh(random_values)

# Plotting the activation functions
plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.plot(random_values, sigmoid_y, label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()


plt.subplot(2, 2, 2)
plt.plot(random_values, relu_y, label='ReLU')
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(random_values, leaky_relu_y, label='Leaky ReLU')
plt.title('Leaky ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(random_values, tanh_y, label='Tanh')
plt.title('Tanh Activation Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.show()
