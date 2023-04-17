from src.gui import *
from src.model import *
from src.layer import *
from src.visualization import *

m = Model([
    Layer(4, LayerActivation.SOFTMAX, LayerInitialization.RANDOM),
    Layer(4, LayerActivation.SOFTMAX, LayerInitialization.RANDOM),
    Layer(4, LayerActivation.SOFTMAX, LayerInitialization.RANDOM),
    Layer(4, LayerActivation.SOFTMAX, LayerInitialization.RANDOM),

], 4)

Visualization(m)