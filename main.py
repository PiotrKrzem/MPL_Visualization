from src.gui import *
from src.model import *
from src.layer import *
from src.visualization import *

m = Model([
    Layer(1, LayerActivation.SOFTMAX, LayerInitialization.RANDOM),
    Layer(16, LayerActivation.SOFTMAX, LayerInitialization.RANDOM)
], 4)

Visualization(m)