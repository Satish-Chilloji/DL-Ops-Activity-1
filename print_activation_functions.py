import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Calculate activation function values
random_values = np.array([-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6])

sigmoid_y = sigmoid(random_values)

# Plotting the activation functions
plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.plot(random_values, sigmoid_y, label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()


plt.tight_layout()
plt.show()
