import numpy as np

from src.layer import Layer
from typing import List

class Model:
    def __init__(self, layers: List[Layer], input_size: int) -> None:
        self.layers = layers
        for idx, layer in enumerate(self.layers):
            layer.__construct__(layers[idx - 1].output_size if idx else input_size)

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        outputs = inputs
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

