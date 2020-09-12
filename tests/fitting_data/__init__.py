from collections import OrderedDict

import numpy as np

NUMBER_OF_RECORDS = 12
COLUMNS_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k"]
COLUMNS = OrderedDict(
    [
        (column_name, np.random.uniform(size=NUMBER_OF_RECORDS))
        for column_name in COLUMNS_NAMES
    ]
)
CONTENT = np.stack(COLUMNS.values(), axis=1).tolist()
ROWS = [list(COLUMNS.keys()), *CONTENT]
VALUES = list(COLUMNS.values())

DEFAULT_SHEET = "default_sheet"
