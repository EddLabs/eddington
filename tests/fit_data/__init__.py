from collections import OrderedDict
import numpy as np

COLUMNS_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k"]
COLUMNS = OrderedDict(
    [(column_name, np.random.uniform(size=12)) for column_name in COLUMNS_NAMES]
)
CONTENT = np.stack(COLUMNS.values(), axis=1).tolist()
ROWS = [list(COLUMNS.keys()), *CONTENT]
VALUES = list(COLUMNS.values())
