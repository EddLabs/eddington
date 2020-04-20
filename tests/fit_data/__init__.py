from collections import OrderedDict
import numpy as np

COLUMNS_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k"]
COLUMNS = OrderedDict(
    [(column_name, np.random.uniform(size=20)) for column_name in COLUMNS_NAMES]
)
