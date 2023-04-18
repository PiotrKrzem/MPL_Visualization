import sys, getopt, time

from domain.layer import LayerActivation, Layer
from domain.model import Model
from typing import List
from input import TrainingData

# -------------- DEFAULT PARAMETERS (IF NOT SPECIFIED BY THE USER) -----------

DEF_HIDDEN_LAYER_NO = 1
DEF_NEURON_NO = [10]
DEF_ACTIVATION_FUNC = LayerActivation.SIGMOID
DEF_BATCH_SIZE = 0
DEF_EPOCHS = 10000
DEF_LEARNING_RATE = 0.01
DEF_MOMENTUM = 0.1
DEF_DATA_SET = TrainingData.LENSES


# ---------------------------- HELPER METHODS ----------------------------------

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
    
    hidden_layers = [Layer(neuron_no[idx], activation_func) for idx in range(hidden_layer_no)]
    output_layer = [Layer(classes_no, LayerActivation.SOFTMAX)]
    layers = [*hidden_layers, *output_layer]

    return Model(layers, input_size)

# Method retrieves activation function by name
#
# Parameters:
# name - name of the activation function
def get_activation_func_by_name(name: str) -> LayerActivation:
    if name == 'SIGMOID':
        return LayerActivation.SIGMOID
    elif name == 'SOFTMAX':
        return LayerActivation.SOFTMAX
    elif name == 'TANH':
        return LayerActivation.TANH
    elif name == 'RELU':
        return LayerActivation.RELU
    
# Method retrieves data set by name
#
# Parameters:
# name - name of the data set
def get_data_set_by_name(name: str) -> TrainingData:
    if name == 'LENSES':
        return TrainingData.LENSES

# ---------------------------- MAIN PROGRAM ----------------------------------

# opts, _ = getopt.getopt(sys.argv[1:], "h:n:f:b:e:l:m:d:")

# for opt, arg in opts:
#     if opt == '-h':
#         DEF_HIDDEN_LAYER_NO = int(arg) 
#         print (f'Number of hidden layers: {arg}')
#     elif opt == '-n':
#         DEF_NEURON_NO = [int(x) for x in arg.split(',')]
#         print (f'Number of neurons per layer: {DEF_NEURON_NO}')
#     elif opt == '-f':
#         DEF_ACTIVATION_FUNC = get_activation_func_by_name(arg)
#         print (f'Activation function: {arg}')
#     elif opt == '-b':
#         DEF_BATCH_SIZE = int(arg)
#         print (f'Number of batches: {arg}')
#     elif opt == '-e':
#         DEF_EPOCHS = int(arg)
#         print (f'Number of epochs: {arg}')
#     elif opt == '-l':
#         DEF_LEARNING_RATE = float(arg)
#         print (f'Learning rate: {DEF_LEARNING_RATE}')
#     elif opt == '-m':
#         DEF_MOMENTUM = float(arg)
#         print (f'Momentum: {DEF_MOMENTUM}')
#     elif opt == '-d':
#         DEF_DATA_SET = get_data_set_by_name(arg)
#         print (f'Data set: {arg}')


input_size, classes_no, inputs, outputs = DEF_DATA_SET.value

model = initialize_model(DEF_HIDDEN_LAYER_NO, DEF_NEURON_NO, DEF_ACTIVATION_FUNC, input_size, classes_no)

time.sleep(3)

if DEF_BATCH_SIZE == 0:
    model.__train_stochastic__(DEF_EPOCHS, inputs, outputs, DEF_LEARNING_RATE, DEF_MOMENTUM)
else:
    model.__train_batches__(DEF_EPOCHS, inputs, outputs, DEF_LEARNING_RATE, DEF_MOMENTUM, DEF_BATCH_SIZE)