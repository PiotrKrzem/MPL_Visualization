from src.layer import Layer
from typing import List

class Model:
    def __init__(self, layers: List[Layer], input_size: int) -> None:
        self.layers = layers

        for idx, layer in enumerate(self.layers):
            layer.__construct__(layers[idx - 1].output_size if idx else input_size)



