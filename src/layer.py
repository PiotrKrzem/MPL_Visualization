import numpy as np

from enum import Enum

class Activation(Enum):
    SIGMOID = lambda input: 1/(1 + np.exp(-input))
    SOFTMAX = lambda input: np.exp(input)/ np.sum(np.exp(input), axis=1, keepdims=True)
    RELU = lambda input: np.multiply(input, input > 0)
    TANH = lambda input: np.tanh(input)

class ActivationDer(Enum):
    SIGMOID = lambda input: Activation.SIGMOID(input) * (1 - Activation.SIGMOID(input))
    SOFTMAX = lambda input: Activation.SOFTMAX(input)
    RELU = lambda input: [0 if val < 0 else 1 for val in input]
    TANH = lambda input: 1 - Activation.TANH(input)**2

class LayerActivation(Enum):
    SIGMOID = lambda input, der: Activation.SIGMOID(input) if not der else ActivationDer.SIGMOID(input)
    SOFTMAX = lambda input, der: Activation.SOFTMAX(input) if not der else ActivationDer.SOFTMAX(input)
    RELU = lambda input, der: Activation.RELU(input) if not der else ActivationDer.RELU(input)
    TANH = lambda input, der: Activation.TANH(input) if not der else ActivationDer.TANH(input)

class Layer:
    def __init__(self, neurons: int, activation: LayerActivation) -> None:
        self.input_size: int
        self.output_size: int = neurons
        self.weights: np.ndarray
        self.biases: np.ndarray
        self.activation = activation

    def __construct__(self, input_size):
        self.input_size = input_size
        self.weights = np.random.sample((self.output_size, self.input_size))
        self.biases = np.random.sample(self.output_size)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.activation(np.dot(self.weights, inputs) + self.biases, False)
    
    def __update_weights__(self, weight_diff: np.ndarray, bias_diff: np.ndarray):
        self.weights = self.weights - weight_diff
        self.biases = self.biases - bias_diff