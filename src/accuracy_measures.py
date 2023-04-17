import numpy as np

# Function computes the number of correct predictions based on expected output
#
# expected_outputs - ground truth for the output
# outputs - predicted outputs
# 
# Returns: float accuracy
def compute_accuracy(expected_outputs: np.ndarray, outputs: np.ndarray) -> float:
    true_values = [np.argmax(expected_outputs[idx]) == output for idx, output in enumerate(outputs)]

    return sum(true_values) / len(outputs)


# Function computes the loss for given epoch and sample
#
# expected_outputs - ground truth for the output for given sample
# outputs - predicted outputs in output layer
# 
# Returns: loss for selected sample
def compute_loss(expected_outputs: np.ndarray, outputs: np.ndarray) -> float:
    output = outputs[-1]
    error = np.dot(expected_outputs, np.log(output))

    return -error