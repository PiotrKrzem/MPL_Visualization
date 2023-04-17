import numpy as np

from os import path
from typing import Tuple
from enum import Enum

DIR_NAME = 'data'
LENSES_CLASSES = 3

def get_file_path(file_name: str) -> str:
    file_absolute_path = path.dirname(__file__)
    return path.join(file_absolute_path, DIR_NAME, file_name)

def get_output_for_row(label: int, classes_no: int) -> np.ndarray:
    output = np.zeros(classes_no)
    output[label - 1] = 1

    return output

def import_lenses_data() -> Tuple[int, int, np.ndarray, np.ndarray]:
    file_path = get_file_path("lenses.data")

    with open(file_path) as file:
        lines = (line for line in file)
        data = np.loadtxt(lines, delimiter=' ', dtype=np.int64)[:,1:]

        inputs = data[:, :-1]
        outputs = np.array([get_output_for_row(label, LENSES_CLASSES) for label in data[:, -1]])

        return len(inputs[0]), LENSES_CLASSES, inputs, outputs

class TrainingData(Enum):
    LENSES = import_lenses_data()