from pytest_cases import THIS_MODULE, parametrize_with_cases

from eddington.print_util import to_precise_string, to_relevant_precision


def case_int_with_2_zeroes():
    inp = dict(a=14, n=2)
    out = dict(relevant_precision=0, precise_string="14.00")
    return inp, out


def case_int_with_4_zeroes():
    inp = dict(a=3, n=4)
    out = dict(relevant_precision=0, precise_string="3.0000")
    return inp, out


def case_negative_int_with_2_zeroes():
    inp = dict(a=-14, n=2)
    out = dict(relevant_precision=0, precise_string="-14.00")
    return inp, out


def case_one_with_3_zeroes():
    inp = dict(a=1, n=3)
    out = dict(relevant_precision=0, precise_string="1.000")
    return inp, out


def case_negative_one_with_3_zeroes():
    inp = dict(a=-1, n=3)
    out = dict(relevant_precision=0, precise_string="-1.000")
    return inp, out


def case_zero_with_3_zeroes():
    inp = dict(a=0, n=3)
    out = dict(relevant_precision=0, precise_string="0.000")
    return inp, out


def case_float_bigger_than_one_reduce_digits():
    inp = dict(a=3.141592, n=2)
    out = dict(relevant_precision=0, precise_string="3.14")
    return inp, out


def case_float_bigger_than_one_add_zeroes():
    inp = dict(a=3.52, n=5)
    out = dict(relevant_precision=0, precise_string="3.52000")
    return inp, out


def case_float_smaller_than_one_reduce_digits():
    inp = dict(a=0.3289, n=1)
    out = dict(relevant_precision=1, precise_string="0.33")
    return inp, out


def case_float_smaller_than_one_add_zeroes():
    inp = dict(a=0.52, n=3)
    out = dict(relevant_precision=1, precise_string="0.5200")
    return inp, out


def case_small_float_reduce_digits():
    inp = dict(a=3.289e-5, n=1)
    out = dict(relevant_precision=5, precise_string="3.3e-05")
    return inp, out


def case_small_float_add_zeroes():
    inp = dict(a=3.289e-5, n=4)
    out = dict(relevant_precision=5, precise_string="3.2890e-05")
    return inp, out


@parametrize_with_cases(argnames="inp, out", cases=THIS_MODULE)
def test_relevant_precision(inp, out):
    _, actual_relevant_precision = to_relevant_precision(inp["a"])
    assert (
        actual_relevant_precision == out["relevant_precision"]
    ), "Relevant precision is different than expected"


@parametrize_with_cases(argnames="inp, out", cases=THIS_MODULE)
def test_precise_string(inp, out):
    assert (
        to_precise_string(inp["a"], inp["n"]) == out["precise_string"]
    ), "Relevant precision is different than expected"
