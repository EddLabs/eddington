from dataclasses import dataclass
import numpy as np


@dataclass
class FitData:

    x: np.ndarray
    xerr: np.ndarray
    y: np.ndarray
    yerr: np.ndarray

    @staticmethod
    def build_from_data_dict(data_dict):
        values = list(data_dict.values())
        return FitData(
            x=np.array(values[0]),
            xerr=np.array(values[1]),
            y=np.array(values[2]),
            yerr=np.array(values[3]),
        )
