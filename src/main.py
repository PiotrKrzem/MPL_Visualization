from layer import LayerActivation, Layer
from model import Model
from typing import List
from input import TrainingData

# DEFAULT PARAMETERS (IF NOT SPECIFIED BY THE USER)

DEF_HIDDEN_LAYER_NO = 2
DEF_NEURON_NO = [10, 10]
DEF_ACTIVATION_FUNC = LayerActivation.SIGMOID
DEF_BATCH_SIZE = 10
DEF_EPOCHS = 1000
DEF_LEARNING_RATE = 0.01
DEF_MOMENTUM = 0.5



def initialize_model(hidden_layer_no: int, 
                     neuron_no: List[int],
                     activation_func: LayerActivation, 
                     input_size: int,
                     classes_no: int) -> Model:
    
    hidden_layers = [Layer(neuron_no[idx], activation_func) for idx in range(hidden_layer_no)]
    output_layer = [Layer(classes_no, activation_func)]
    layers = [*hidden_layers, *output_layer]

    return Model(layers, input_size)


input_size, classes_no, inputs, outputs = TrainingData.LENSES.value

model = initialize_model(DEF_HIDDEN_LAYER_NO, DEF_NEURON_NO, DEF_ACTIVATION_FUNC, input_size, classes_no)
model.__train__(DEF_EPOCHS, inputs, outputs, DEF_LEARNING_RATE, DEF_MOMENTUM)