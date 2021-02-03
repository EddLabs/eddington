from collections import OrderedDict

import numpy as np

from eddington.statistics import Statistics

NUMBER_OF_RECORDS = 12
COLUMNS_NAMES = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "k"]
NUMBER_OF_COLUMNS = len(COLUMNS_NAMES)
COLUMNS = OrderedDict(
    [
        (column_name, np.random.uniform(size=NUMBER_OF_RECORDS))
        for column_name in COLUMNS_NAMES
    ]
)
CONTENT = np.stack(list(COLUMNS.values()), axis=1).tolist()
ROWS = [list(COLUMNS.keys()), *CONTENT]
VALUES = list(COLUMNS.values())
STATISTICS = {
    column: Statistics.from_array(values) for column, values in COLUMNS.items()
}
