import numpy as np

from enum import Enum

class LayerActivation(Enum):
    SIGMOID = lambda input: 1/(1 + np.exp(-input))
    SOFTMAX = lambda input: np.exp(input)/ np.sum(np.exp(input), axis=1, keepdims=True)
    RELU = lambda input: np.multiply(input, input > 0)
    TANH = lambda input: np.tanh(input)

class LayerInitialization(Enum):
    RANDOM = lambda shape: np.random.rand(*shape)
    GLOROT = lambda shape: None # TODO

class Layer:
    ID = 0
    def __init__(self, neurons: int, activation: LayerActivation, initialization: LayerInitialization) -> None:
        self.input_size: int
        self.output_size: int = neurons
        self.weights: np.ndarray
        self.biases: np.ndarray
        self.activation = activation
        self.initialization = initialization

        self.id = Layer.ID
        Layer.ID += 1

    def __construct__(self, input_size):
        self.input_size = input_size
        self.weights = self.initialization((self.input_size, self.output_size))
        self.biases = self.initialization((self.output_size, 1))

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.activation(np.dot(self.weights, inputs) + self.biases)