import time
from typing import List

from src.argparser import parse_args
from src.domain.model import Model
from src.domain.layer import LayerActivation, Layer, InputLayer

# ----------------------- HELPER METHODS ------------------------------

# Method initialize the model based on the passed parameters
#
# Parameters:
# hidden_layer_no - number of hidden layers
# neuron_no       - number of neurons used per layer
# activation_func - activation function used in the hidden layers
# input_size      - size of the input data
# classes_no      - number of classes in the data
def initialize_model(hidden_layer_no: int, 
                    neuron_no: List[int],
                    activation_func: LayerActivation, 
                    input_size: int,
                    classes_no: int) -> Model:

    input_layer = [InputLayer(input_size)]
    hidden_layers = [Layer(neuron_no[idx], activation_func) for idx in range(hidden_layer_no)]
    output_layer = [Layer(classes_no, LayerActivation.SOFTMAX)]
    layers = [*input_layer, *hidden_layers, *output_layer]

    return Model(layers)

# ---------------------------- MAIN PROGRAM ----------------------------------

def main() -> None: 
    arguments = parse_args()
    input_size, classes_no, inputs, outputs = arguments.dataset.value

    model = initialize_model(arguments.hidden_layers_no, 
                             arguments.neurons_no, 
                             arguments.activation_func, 
                             input_size, 
                             classes_no
    )

    time.sleep(3)

    if arguments.batch_size == 0:
        model.__train_stochastic__(arguments.epochs, inputs, outputs, arguments.learning_rate, arguments.momentum, arguments.early_stopping)
    else:
        model.__train_batches__(arguments.epochs, inputs, outputs, arguments.learning_rate, arguments.momentum, arguments.early_stopping, arguments.batch_size)

if __name__ == "__main__":
    main()