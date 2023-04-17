import numpy as np

from tqdm import tqdm
from sys import stdout
from time import sleep
from signal import signal, SIGINT
from dataclasses import dataclass
from threading import Thread, Event
from typing import Any, List, Tuple, Dict

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec as GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec as Grid

from src.layer import Layer
from src.model import Model
from src.arrow3D import Arrow3D

from random import seed as r_seed, randint
from numpy.random import seed as n_seed
r_seed(69), n_seed(69)

MODEL_DEPTH_PX = 1000
MODEL_NEURONS_DISTANCE_PX = 10

@dataclass
class NeuronMarkerData:
    xyz: List[int]
    layer_idx: int
    neuron_idx: int
    marker: Any # I'm lazy

@dataclass
class ConnectionMarkerData:
    src_layer_idx: int
    dst_layer_idx: int
    src_neuron_idx: int
    dst_neuron_idx: int
    marker: Any # I'm lazy

class Visualization:
    def __init__(self, model: Model) -> None:
        self.model = model

        self.prepare_interface()
        self.prepare_markers()
        plt.show()

    def prepare_interface(self):
        self.figure = plt.figure("Visualization", figsize=(10,10))
        self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.figure.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.figure.canvas.mpl_connect('close_event', self.on_exit)

        self.axis = self.figure.add_subplot(111, projection='3d')
        self.axis.spines['bottom'].set(color = 'white', linewidth = 3)
        self.axis.spines['top'].set(color = 'white', linewidth = 3)
        self.axis.spines['left'].set(color = 'white', linewidth = 3)
        self.axis.spines['right'].set(color = 'white', linewidth = 3)
        self.axis.set_xlabel('X Label')
        self.axis.set_ylabel('Y Label')
        self.axis.set_zlabel('Z Label')
        self.figure.add_subplot(self.axis)

    def prepare_markers(self):
        layer_width_div = MODEL_DEPTH_PX // (len(self.model.layers) - 1)

        self.neuron_markers: List[List[NeuronMarkerData]] = []
        for layer_idx, layer in enumerate(self.model.layers):
            approx_side_length = np.log2(layer.output_size)
            side_length = int(approx_side_length) + (1 if approx_side_length > int(approx_side_length) else 0)
            side_length = 1 if layer.output_size <= 3 else side_length
            xy_start_value = -MODEL_NEURONS_DISTANCE_PX * side_length // 2

            layer_neuron_markers = []
            layer_connection_markers = []

            for neuron_idx in range(side_length*side_length):
                if neuron_idx >= layer.output_size: break

                x = xy_start_value + (neuron_idx % side_length) * MODEL_NEURONS_DISTANCE_PX
                z = xy_start_value + (neuron_idx // side_length) * MODEL_NEURONS_DISTANCE_PX
                y = layer_width_div * layer_idx

                neuron_square_marker = self.axis.scatter(
                    [x], [y], [z],
                    marker='s', color ='red', zorder = 2, s = 64
                )
                layer_neuron_markers.append(
                    NeuronMarkerData(
                        (x, y, z), 
                        layer_idx, 
                        neuron_idx, 
                        neuron_square_marker
                    )
                )
            self.neuron_markers.append(layer_neuron_markers)

        self.connections_markers: List[List[ConnectionMarkerData]] = []
        for neuron_layer_idx in range(1,len(self.neuron_markers)):
            for dst_neuron in self.neuron_markers[neuron_layer_idx]:
                for src_neuron in self.neuron_markers[neuron_layer_idx - 1]:
                    src_x, src_y, src_z = src_neuron.xyz
                    dst_x, dst_y, dst_z = dst_neuron.xyz
                    dx = dst_x - src_x
                    dy = dst_y - src_y
                    dz = dst_z - src_z
                    connection_line_marker = self.axis.quiver(
                        [src_x], [src_y], [src_z],
                        [dx], [dy], [dz],
                        color = 'red', lw = 0.2,
                    )

                    layer_connection_markers.append(
                        ConnectionMarkerData(
                            neuron_layer_idx - 1, 
                            neuron_layer_idx, 
                            src_neuron.neuron_idx,
                            dst_neuron.neuron_idx, 
                            connection_line_marker
                        )
                    )
            self.connections_markers.append(layer_connection_markers)

    def on_key_press(self, event):
        stdout.flush()
        if event.key == 'x':
            self.do_one_random_walk()
        elif event.key == 'c':
            self.start_auto_walk()
        elif event.key == 'v':
            self.stop_auto_walk()
        elif event.key == 'z':
            self.toggle_rendering()

    def on_mouse_press(self, event):
        return

    def on_exit(self, event):
        return