from src.domain.model import *
from src.domain.layer import *
from src.visualization import *

m = Model([
    InputLayer(4),
    Layer(16, LayerActivation.SOFTMAX),
    Layer(16, LayerActivation.SOFTMAX),
    Layer(16, LayerActivation.SOFTMAX),
    Layer(4, LayerActivation.SOFTMAX),
])

Visualization(m)