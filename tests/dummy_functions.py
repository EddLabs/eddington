from eddington_core import fit_function, fit_function_generator


@fit_function(n=2, save=False)
def dummy_func1(a, x):
    return a[0] + a[1] * x ** 2


@fit_function_generator(
    name="generator_with_2_parameters",
    parameters=["p0", "p1"],
    syntax="Some syntax",
    save=False,
)
def dummy_generator_with_2_parameters(p0, p1):
    @fit_function(n=2, save=False)
    def dummy_function(a, x):
        return p0 * a[0] + p1 * a[1] * x

    return dummy_function


@fit_function_generator(
    name="generator_with_1_parameter",
    parameters="p1",
    syntax="Some syntax",
    save=False,
)
def dummy_generator_with_1_parameter(p1):
    @fit_function(n=2, save=False)
    def dummy_function(a, x):
        return a[0] + p1 * a[1] * x

    return dummy_function
