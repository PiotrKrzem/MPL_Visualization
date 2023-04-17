import numpy as np

from layer import Layer
from typing import List, Tuple

def compute_accuracy(expected_outputs: np.ndarray, outputs: np.ndarray) -> float:
    count = 0

    for idx, output in enumerate(outputs):
        expected = np.argmax(expected_outputs[idx]) + 1
        count += 1 if expected == output else 0

    return count / len(outputs)


class Model:
    def __init__(self, layers: List[Layer], input_size: int) -> None:
        self.layers = layers

        for idx, layer in enumerate(self.layers):
            layer.__construct__(layers[idx - 1].output_size if idx else input_size)

    def __forward__(self, input_row: np.ndarray) -> np.ndarray:
        input = input_row
        outputs = []

        for layer in self.layers:
            input = layer.__call__(input)
            outputs.append(input)

        return outputs

    def __backward__(self, outputs: List[List[int]], expected_output: np.ndarray, learn_rate: float, momentum: float) -> float:
        error = None

        for idx, layer in reversed(list(enumerate(self.layers))):
            activation = layer.activation
            output = outputs[idx]
            input = output[idx - 1]

            if error is None:
                error = (np.array(output) - expected_output)
                loss = (error**2).sum()
            else:
                error = np.dot(error, self.layers[idx + 1].weights)

            delta = error * activation(output, der=True)
            weights_diff = learn_rate * np.kron(delta, input)
            bias_diff = learn_rate * delta

            layer.__update_weights__(weights_diff, bias_diff)

        return loss
    

    def __train__(self, epochs: int, inputs: np.ndarray, expected_outputs: np.ndarray, learn_rate: float, momentum: float):
        sample_size = len(inputs)

        for epoch in range(epochs):
            loss = 0
            results = []

            for idx in range(sample_size):
                inputs_sample = inputs[idx]
                expected_output = expected_outputs[idx]

                outputs = self.__forward__(inputs_sample)
                loss += self.__backward__(outputs, expected_output, learn_rate, momentum)

                results.append(np.argmax(outputs[-1]) + 1)

            accuracy = compute_accuracy(expected_outputs, results)
            print(f"Epoch: {epoch}, Accuracy: {accuracy}, Loss: {loss}")

