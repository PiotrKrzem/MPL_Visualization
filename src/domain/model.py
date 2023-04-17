import numpy as np

from domain.layer import Layer
from typing import List
from accuracy_measures import compute_accuracy, compute_loss


class Model:
    def __init__(self, layers: List[Layer], input_size: int) -> None:
        self.layers = layers

        for idx, layer in enumerate(self.layers):
            layer.__construct__(layers[idx - 1].output_size if idx else input_size)

    def __forward__(self, input_row: np.ndarray) -> np.ndarray:
        input = input_row
        outputs = [input_row]

        for layer in self.layers:
            input = layer.__call__(input)
            outputs.append(input)

        return outputs

    def __backward__(self, outputs: List[List[int]], 
                     expected_output: np.ndarray, 
                     learn_rate: float, 
                     momentum: float,
                     prev_weights_diffs: List[np.ndarray]) -> List[np.ndarray]:
        error = None
        weights_diffs = []

        for idx, layer in reversed(list(enumerate(self.layers))):
            activation = layer.activation
            output = outputs[idx + 1]
            input = outputs[idx]
            prev_weights_diff = prev_weights_diffs[idx] if prev_weights_diffs else 0

            if error is None:
                error = (output - expected_output)
            else:
                error = np.dot(error, self.layers[idx + 1].weights)

            delta = error * activation(output, der=True)
        
            weights_diff = learn_rate * np.kron(delta, input) + prev_weights_diff * momentum
            bias_diff = learn_rate * delta

            layer.__update_weights__(weights_diff, bias_diff)
            weights_diffs.append(weights_diff)

        return weights_diffs[::-1]
    

    def __train__(self, epochs: int, inputs: np.ndarray, expected_outputs: np.ndarray, learn_rate: float, momentum: float):
        sample_size = len(inputs)

        for epoch in range(epochs):
            total_loss = 0
            results = []
            weight_diffs = None

            for idx in range(sample_size):
                inputs_sample = inputs[idx]
                expected_output = expected_outputs[idx]

                outputs = self.__forward__(inputs_sample)
                weight_diffs = self.__backward__(outputs, expected_output, learn_rate, momentum, weight_diffs)
                
                total_loss += compute_loss(expected_output, outputs)
                results.append(np.argmax(outputs[-1]))

            accuracy = compute_accuracy(expected_outputs, results)
            print(f"Epoch: {epoch}, Accuracy: {accuracy}, Loss: {total_loss}")

