import math


def to_relevant_precision(a):
    if a == 0:
        return 0, 0
    precision = 0
    abs_a = math.fabs(a)
    while abs_a < 1.0:
        abs_a *= 10
        precision += 1
    if a < 0:
        return -abs_a, precision
    return abs_a, precision


def to_precise_string(a, n):
    new_a, precision = to_relevant_precision(a)
    if precision < 3:
        return f"{a:.{n + precision}f}"
    return f"{new_a:.{n}f}e-0{precision}"
