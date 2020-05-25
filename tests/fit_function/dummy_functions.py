from eddington_core import fit_function


@fit_function(n=2, syntax="a[0] + a[1] * x ** 2", save=False)
def dummy_func1(a, x):
    return a[0] + a[1] * x ** 2
