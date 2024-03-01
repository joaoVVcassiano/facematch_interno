from typing import Union
import numpy as np

def findCosineDistance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> np.float64:
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    a = np.vdot(source_representation, test_representation)
    b = np.sum(np.dot(source_representation, source_representation))
    c = np.sum(np.dot(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
