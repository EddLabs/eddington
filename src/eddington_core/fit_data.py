from dataclasses import dataclass
import numpy as np


@dataclass
class FitData:

    x: np.ndarray
    xerr: np.ndarray
    y: np.ndarray
    yerr: np.ndarray
