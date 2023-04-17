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
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.pyplot import cm
from matplotlib.collections import PathCollection
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec as GridSpec
from matplotlib.gridspec import GridSpecFromSubplotSpec as Grid

from src.layer import Layer
from src.model import Model
from src.arrow3D import Arrow3D

from random import seed as r_seed
from numpy.random import seed as n_seed
r_seed(69), n_seed(69)

MODEL_DEPTH_PX = 40
MODEL_NEURONS_DISTANCE_PX_FIRST_LAST_LAYER = 5
MODEL_NEURONS_DISTANCE_PX = 20
NOISE_AMOUNT = MODEL_NEURONS_DISTANCE_PX // 10

@dataclass
class NeuronMarkersData:
    xyz: List[np.ndarray]
    layer: Layer
    marker: PathCollection

@dataclass
class ConnectionMarkerData:
    layer: Layer
    marker: Line3DCollection

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

        self.axis:Axes = self.figure.add_subplot(111, projection='3d')
        self.axis.spines['bottom'].set(color = 'white', linewidth = 0)
        self.axis.spines['top'].set(color = 'white', linewidth = 0)
        self.axis.spines['left'].set(color = 'white', linewidth = 0)
        self.axis.spines['right'].set(color = 'white', linewidth = 0)
        # self.axis.imshow([[0.2, 0.3], [0.8, 0.9]], interpolation='bicubic', cmap=cm.copper, alpha=1)
        # self.axis.set_xlabel('X Label')
        # self.axis.set_ylabel('Y Label')
        # self.axis.set_zlabel('Z Label')
        # self.axis.set_xlim([-1, 8])
        # self.axis.set_ylim([-1, 8])
        # self.axis.set_zlim([-1, 8])
        self.axis.grid(False)
        self.axis.set_xticks([])
        self.axis.set_yticks([])
        self.axis.set_zticks([])
        self.axis.set_axis_off()

    def prepare_markers(self):
        layer_width_div = MODEL_DEPTH_PX // (len(self.model.layers) - 1)

        self.layers_neurons_data: List[NeuronMarkersData] = []
        for layer_idx, layer in enumerate(self.model.layers):
            if layer_idx == 0 or layer_idx == len(self.model.layers) - 1:
                side_length = layer.output_size
                x_start_value = int(-MODEL_NEURONS_DISTANCE_PX_FIRST_LAST_LAYER * side_length / 2)
                x_grid = np.linspace(0, MODEL_NEURONS_DISTANCE_PX_FIRST_LAST_LAYER * side_length, side_length) + x_start_value
                y_grid = np.full((x_grid.shape[0]), layer_width_div * layer_idx)
                z_grid = np.zeros((x_grid.shape[0]))
            else:
                approx_side_length = np.log2(layer.output_size)
                approx_side_length_incr = 1 if approx_side_length > int(approx_side_length) else 0
                side_length = int(approx_side_length) + approx_side_length_incr
                xz_start_value = -MODEL_NEURONS_DISTANCE_PX * side_length // 2
                x_range = np.linspace(0, MODEL_NEURONS_DISTANCE_PX * side_length, side_length) + xz_start_value
                z_range = np.linspace(0, MODEL_NEURONS_DISTANCE_PX * side_length, side_length) + xz_start_value
                x_grid, z_grid = np.meshgrid(x_range, z_range)
                x_grid, z_grid = x_grid.flatten(), z_grid.flatten()
                y_grid = np.full((x_grid.shape[0]), layer_width_div * layer_idx)

                x_noise = np.random.randint(-NOISE_AMOUNT, NOISE_AMOUNT + 1, size = x_grid.shape)
                y_noise = np.random.randint(-NOISE_AMOUNT, NOISE_AMOUNT + 1, size = x_grid.shape)
                z_noise = np.random.randint(-NOISE_AMOUNT, NOISE_AMOUNT + 1, size = x_grid.shape)
                x_grid += x_noise
                y_grid += y_noise
                z_grid += z_noise

            neuron_markers = self.axis.scatter(
                x_grid, y_grid, z_grid,
                marker='s', zorder = 2, s = 64,
                color='black', alpha=1
            )
            self.layers_neurons_data.append(
                NeuronMarkersData(
                    (x_grid, y_grid, z_grid),
                    layer,
                    neuron_markers
                )
            )

        self.layers_connections_data: List[ConnectionMarkerData] = []
        for layer_idx, dst_neurons_data in enumerate(self.layers_neurons_data):
            if not layer_idx: continue

            x,y,z = dst_neurons_data.xyz
            src_neurons_data = self.layers_neurons_data[layer_idx - 1]
            dst_x = np.tile(x, src_neurons_data.layer.output_size)
            dst_y = np.tile(y, src_neurons_data.layer.output_size)
            dst_z = np.tile(z, src_neurons_data.layer.output_size)

            tile_amount = x.shape[0]
            x,y,z = src_neurons_data.xyz
            src_x = np.tile(x[np.newaxis].T, (1,tile_amount)).flatten()
            src_y = np.tile(y[np.newaxis].T, (1,tile_amount)).flatten()
            src_z = np.tile(z[np.newaxis].T, (1,tile_amount)).flatten()

            delta_x = dst_x - src_x
            delta_y = dst_y - src_y
            delta_z = dst_z - src_z

            connection_markers = self.axis.quiver(
                src_x, src_y, src_z,
                delta_x, delta_y, delta_z,
                arrow_length_ratio = 0,
                linewidths = 0.4
            )
            self.layers_connections_data.append(
                ConnectionMarkerData(
                    layer,
                    connection_markers
                )
            )

    def update(self):
        output = np.random.rand(self.model.layers[0].output_size)
        for layer_idx, layer in enumerate(self.model.layers):
            output_size = layer.output_size
            weights_size = np.multiply(*layer.weights.shape)
            output = np.random.rand(output_size) #TODO REMOVE
            # if layer_idx:
            #     output = layer(output)
            # else:
            #     blablabla

            color = np.ones((output_size, 3)) * np.tile(output[np.newaxis].T, 3)
            # color = np.maximum(color, 0.5)
            alpha = np.ones((output_size, 1))
            color_rgba = np.hstack((color, alpha))
            self.layers_neurons_data[layer_idx].marker.set_color(color_rgba)

            # weights = layer.weights.flatten()
            # color = np.tile(np.array([[0,0,1]]), (weights_size, 1)) * np.tile(weights[np.newaxis].T, 3)
            # alpha = np.ones((weights_size, 1))
            # color_rgba = np.hstack((color, alpha))
            # self.layers_connections_data[layer_idx - 1].marker.set_color(color_rgba)

            if not layer_idx: continue
            

        self.figure.canvas.draw()

    def on_key_press(self, event):
        stdout.flush()
        if event.key == 'x':
            self.update()
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